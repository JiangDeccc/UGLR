"""Pairwise statistical prior and exact joint decoding for RPG attributes."""

from itertools import combinations

import torch


ATTR_ORDER = ("gender", "education", "age", "income")


def _pair_key(left, right):
    return f"{left}__{right}"


class PairwiseAttributePrior:
    """Six smoothed pairwise compatibility tables estimated from train labels only."""

    def __init__(self, num_classes, tables, counts, num_samples,
                 alpha=1.0, mode="pmi", clip=3.0, attr_order=ATTR_ORDER):
        self.attr_order = tuple(attr_order)
        self.num_classes = {attr: int(num_classes[attr]) for attr in self.attr_order}
        self.tables = tables
        self.counts = counts
        self.num_samples = int(num_samples)
        self.alpha = float(alpha)
        self.mode = mode
        self.clip = None if clip is None else float(clip)

    @classmethod
    def fit(cls, labels, num_classes, alpha=1.0, mode="pmi", clip=3.0,
            attr_order=ATTR_ORDER):
        """Estimate pairwise log-joint or PMI tables from fully-known labels.

        ``labels`` must contain only class ids supported by the model. Callers should
        pass the same unknown-filtered training labels used to fit the classifier.
        """
        if mode not in {"pmi", "joint"}:
            raise ValueError(f"prior mode must be 'pmi' or 'joint', got {mode!r}")
        if alpha <= 0:
            raise ValueError(f"prior alpha must be positive, got {alpha}")
        if clip is not None and clip <= 0:
            raise ValueError(f"prior clip must be positive or None, got {clip}")
        if not labels:
            raise ValueError("cannot fit pairwise prior from an empty label set")

        attr_order = tuple(attr_order)
        class_counts = {attr: int(num_classes[attr]) for attr in attr_order}
        counts = {}
        for left, right in combinations(attr_order, 2):
            counts[_pair_key(left, right)] = torch.zeros(
                class_counts[left], class_counts[right], dtype=torch.float64
            )

        for row_index, label in enumerate(labels):
            values = {}
            for attr in attr_order:
                if attr not in label:
                    raise KeyError(f"training label row {row_index} is missing {attr!r}")
                value = int(label[attr])
                if not 0 <= value < class_counts[attr]:
                    raise ValueError(
                        f"training label row {row_index} has unsupported {attr}={value}; "
                        f"expected [0, {class_counts[attr] - 1}]"
                    )
                values[attr] = value
            for left, right in combinations(attr_order, 2):
                counts[_pair_key(left, right)][values[left], values[right]] += 1

        tables = {}
        for left, right in combinations(attr_order, 2):
            key = _pair_key(left, right)
            smoothed = counts[key] + float(alpha)
            joint = smoothed / smoothed.sum()
            if mode == "joint":
                compatibility = joint.log()
            else:
                left_marginal = joint.sum(dim=1, keepdim=True)
                right_marginal = joint.sum(dim=0, keepdim=True)
                compatibility = (joint / (left_marginal * right_marginal)).log()
            # PMI can become extreme for rare cells; raw log-joint is already <= 0
            # and should not be flattened by the PMI-oriented default clip=3.
            if clip is not None and mode == "pmi":
                compatibility = compatibility.clamp(min=-clip, max=clip)
            tables[key] = compatibility.to(torch.float32)

        return cls(
            num_classes=class_counts,
            tables=tables,
            counts=counts,
            num_samples=len(labels),
            alpha=alpha,
            mode=mode,
            clip=clip,
            attr_order=attr_order,
        )

    def candidate_prior_scores(self, candidates):
        """Return C(u) for candidates ordered according to ``attr_order``."""
        if candidates.ndim != 2 or candidates.size(1) != len(self.attr_order):
            raise ValueError(
                f"candidates must have shape [K, {len(self.attr_order)}], "
                f"got {tuple(candidates.shape)}"
            )
        attr_column = {attr: index for index, attr in enumerate(self.attr_order)}
        scores = torch.zeros(candidates.size(0), dtype=torch.float32)
        cpu_candidates = candidates.cpu()
        for left, right in combinations(self.attr_order, 2):
            table = self.tables[_pair_key(left, right)].cpu()
            scores += table[
                cpu_candidates[:, attr_column[left]],
                cpu_candidates[:, attr_column[right]],
            ]
        return scores

    def to_dict(self, include_tables=True):
        result = {
            "attr_order": list(self.attr_order),
            "num_classes": self.num_classes.copy(),
            "num_samples": self.num_samples,
            "alpha": self.alpha,
            "mode": self.mode,
            "clip": self.clip,
        }
        if include_tables:
            result["pairs"] = {
                key: {
                    "counts": self.counts[key].tolist(),
                    "compatibility": self.tables[key].tolist(),
                }
                for key in self.tables
            }
        return result


class JointAttributeDecoder:
    """Exact argmax over all attribute tuples under unary + pairwise scores."""

    def __init__(self, prior, beta=0.1):
        if beta < 0:
            raise ValueError(f"joint decoding beta must be non-negative, got {beta}")
        self.prior = prior
        self.beta = float(beta)
        axes = [torch.arange(prior.num_classes[attr]) for attr in prior.attr_order]
        self.candidates = torch.cartesian_prod(*axes).to(torch.long)
        self.prior_scores = prior.candidate_prior_scores(self.candidates)

    @torch.no_grad()
    def decode(self, logits_map, beta=None, return_scores=False):
        """Decode a batch and return ``{attr: LongTensor[B]}`` predictions."""
        beta = self.beta if beta is None else float(beta)
        if beta < 0:
            raise ValueError(f"joint decoding beta must be non-negative, got {beta}")

        first_logits = logits_map[self.prior.attr_order[0]]
        batch_size = first_logits.size(0)
        device = first_logits.device
        candidates = self.candidates.to(device)
        unary_scores = torch.zeros(
            batch_size, candidates.size(0), device=device, dtype=torch.float32
        )

        for column, attr in enumerate(self.prior.attr_order):
            logits = logits_map[attr]
            if logits.ndim != 2 or logits.size(0) != batch_size:
                raise ValueError(f"invalid logits shape for {attr}: {tuple(logits.shape)}")
            expected_classes = self.prior.num_classes[attr]
            if logits.size(1) != expected_classes:
                raise ValueError(
                    f"{attr} logits have {logits.size(1)} classes; "
                    f"prior expects {expected_classes}"
                )
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            unary_scores += log_probs[:, candidates[:, column]]

        scores = unary_scores + beta * self.prior_scores.to(device).unsqueeze(0)
        best_indices = scores.argmax(dim=1)
        best_candidates = candidates[best_indices]
        predictions = {
            attr: best_candidates[:, column]
            for column, attr in enumerate(self.prior.attr_order)
        }
        if not return_scores:
            return predictions
        return predictions, {
            "best_total_score": scores.gather(1, best_indices.unsqueeze(1)).squeeze(1),
            "best_unary_score": unary_scores.gather(
                1, best_indices.unsqueeze(1)
            ).squeeze(1),
            "best_prior_score": self.prior_scores.to(device)[best_indices],
        }

    def to_dict(self, include_tables=True):
        return {
            "beta": self.beta,
            "num_candidates": int(self.candidates.size(0)),
            "prior": self.prior.to_dict(include_tables=include_tables),
        }
