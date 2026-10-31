"""Final Amazon and ML1M profile models: margin evidence and sqrt-Gini final-query gating."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MOVIELENS_ATTRS = ("gender", "occupation", "age")

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class AttrRetrievalHead(nn.Module):
    'Cosine codebook retrieval with a learnable log inverse temperature.\n\n    Normalize queries and keys before scoring. Clamp log_logit_scale before\n    exponentiation, and exclude codebooks and temperature scalars from weight\n    decay. Parameter groups are configured by the training scripts.\n    '


    LOGIT_SCALE_MAX = 100.0

    def __init__(
        self,
        hidden_size,
        num_classes,
        use_cosine=True,
        temperature=0.07,
        learnable_temperature=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.code_dim = hidden_size
        self.use_cosine = use_cosine
        self.learnable_temperature = learnable_temperature


        self.query_proj = ResBlock(hidden_size)


        self.codebook = nn.Embedding(num_classes, hidden_size)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)


        if use_cosine:

            init = math.log(1.0 / temperature)
            init_t = torch.tensor(init, dtype=torch.float32)
            if learnable_temperature:
                self.log_logit_scale = nn.Parameter(init_t)
            else:
                self.register_buffer("log_logit_scale", init_t)
        else:
            init_t = torch.tensor(math.log(temperature), dtype=torch.float32)
            if learnable_temperature:
                self.log_tau = nn.Parameter(init_t)
            else:
                self.register_buffer("log_tau", init_t)

    def _logit_scale(self):
        'Clamp the log-scale parameter before exponentiation to preserve its gradient.'
        log_scale = self.log_logit_scale.clamp(max=math.log(self.LOGIT_SCALE_MAX))
        return log_scale.exp()

    def forward(self, s):
        'Input s: [B, hidden_size], the final token state. Return FP32 logits: [B, num_classes].'
        q = self.query_proj(s)                         # [B, code_dim]
        codebook = self.codebook.weight                # [num_classes, code_dim]


        q = q.float()
        codebook = codebook.float()

        if self.use_cosine:

            q = F.normalize(q, dim=-1)
            codebook = F.normalize(codebook, dim=-1)
            logits = self._logit_scale() * torch.matmul(q, codebook.t())
        else:
            tau = torch.clamp(self.log_tau.exp(), min=1e-3)
            logits = torch.matmul(q, codebook.t()) / tau

        return logits


class MultiTaskLossWeighting(nn.Module):
    def __init__(self, num_tasks=4, mode="uncertainty", fixed_weights=None):
        super().__init__()
        assert mode in ("uncertainty", "fixed")
        self.mode = mode
        self.num_tasks = num_tasks

        if mode == "uncertainty":

            self.log_var = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        else:
            if fixed_weights is None:
                fixed_weights = [1.0] * num_tasks
            self.register_buffer(
                "fixed_weights",
                torch.tensor(fixed_weights, dtype=torch.float32),
            )

    def forward(self, task_losses):
        'Combine num_tasks scalar losses; return total_loss and weight_info_dict.'
        losses = torch.stack([l.float() for l in task_losses])   # [T]

        if self.mode == "uncertainty":
            precision = torch.exp(-self.log_var)                 # 1/σ^2
            total = (precision * losses + 0.5 * self.log_var).sum()
            info = {
                "sigma": torch.exp(0.5 * self.log_var).detach(),
                "weight": precision.detach(),
            }
        else:
            w = self.fixed_weights.to(losses.device)
            total = (w * losses).sum()
            info = {"weight": w.detach()}

        return total, info


def build_ordinal_soft_labels(targets, num_classes, sigma=1.0, device=None):
    'Map integer targets [B] to Gaussian soft labels [B, num_classes].'
    if device is None:
        device = targets.device
    idx = torch.arange(num_classes, device=device).float().unsqueeze(0)   # [1, C]
    center = targets.float().unsqueeze(1)                                 # [B, 1]
    dist2 = (idx - center) ** 2
    logits = -dist2 / (2.0 * sigma * sigma)
    soft = torch.softmax(logits, dim=-1)
    return soft


def soft_cross_entropy(logits, soft_targets):
    """logits: [B, C]（float）, soft_targets: [B, C]"""
    log_prob = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_prob).sum(dim=-1).mean()


class DirectQueryBlock(nn.Module):
    """Produce an attribute-specific query without an identity skip connection.

    The baseline ResBlock zero-initializes its linear layer because the skip
    connection supplies a valid initial query.  That initialization cannot be
    reused here: without the skip it would make every initial query exactly
    zero.  Xavier initialization keeps the direct query nonzero from the first
    forward pass while remaining symmetric across attributes.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(x))


class _ProfileModel(nn.Module):
    """Shared final-model operations; no alternative model variants."""
    def set_mixed_precision(self, device, backbone_dtype=torch.bfloat16):
        self._base_set_mixed_precision_heads(device, backbone_dtype)
        for name, module in self.named_modules():
            if name.startswith('evidence_'):
                module.to(device=device, dtype=torch.float32)
        return self

    def dtype_info(self):
        info = self._base_dtype_info_heads()
        info['evidence_memory_proj'] = str(self.evidence_memory_proj.weight.dtype)
        info['evidence_attention'] = str(self.evidence_query_proj.weight.dtype)
        return info

    def encode_backbone(self, input_ids, attention_mask):
        """Return the final hidden state sequence produced by the backbone.

        Kept as a separate method so backbones whose causal-LM wrapper has a
        different internal layout (for example Qwen3.5's multimodal wrapper)
        can reuse the RPG heads without duplicating the full forward pass.
        """
        outputs = self.backbone.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def _item_memory(self, hidden, positions, item_mask):
        safe_positions = positions.clamp(min=0, max=hidden.size(1) - 1)
        gathered = hidden.gather(1, safe_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        # Detach *before* both trainable memory layers. Only selected item states are cast to FP32.
        return self.evidence_memory_proj(self.evidence_memory_norm(gathered.detach().float()))

    def _attention(self, q0, memory, item_mask):
        valid = item_mask.bool().any(-1)
        if not valid.any():
            return q0.new_zeros((*q0.shape[:2], self.evidence_dim), dtype=torch.float32)
        queries = self.evidence_query_proj(self.evidence_query_norm(q0.float()))
        keys, values = self.evidence_key_proj(memory), self.evidence_value_proj(memory)
        scores = torch.einsum('btd,bnd->btn', queries, keys) / math.sqrt(self.evidence_dim)
        context = torch.zeros_like(queries)
        valid_scores = scores[valid].masked_fill(~item_mask[valid, None, :].bool(), float('-inf'))
        weights = F.softmax(valid_scores, dim=-1)
        context[valid] = torch.einsum('btn,bnd->btd', weights, values[valid])
        return context

    def _final_queries(self, q0, context, rho, item_mask):
        queries = []
        has_items = item_mask.bool().any(-1, keepdim=True).float()
        for index, attr in enumerate(self.attr_order):
            delta = self.evidence_output_projs[attr](F.layer_norm(context[:, index], (self.evidence_dim,)))
            bounded = (delta / delta.norm(dim=-1, keepdim=True).clamp_min(1.)) * has_items
            # The q0 norm is deliberately detached: residual magnitude cannot evade rho.
            queries.append(q0[:, index].float() + rho[:, index:index + 1] *
                           q0[:, index].float().norm(dim=-1, keepdim=True).detach() * bounded)
        return torch.stack(queries, dim=1)

    @torch.no_grad()
    def gini_uncertainty(self, logits):
        scores = []
        for attr in self.attr_order:
            probabilities = torch.softmax(logits[attr].float(), dim=-1)
            classes = probabilities.size(-1)
            if classes < 2:
                raise ValueError('Gini gating requires at least two classes per attribute')
            # Algebraically 1 - normalized_gini; avoid cancellation near uniform p.
            confidence_squared = (probabilities - 1 / classes).square().sum(-1) / (1 - 1 / classes)
            scores.append(1 - confidence_squared.clamp(0, 1).sqrt())
        return torch.stack(scores, dim=-1)

    def _retrieve(self, attr, query):
        head = getattr(self, f'{attr}_head')
        q, codebook = query.float(), head.codebook.weight.float()
        if head.use_cosine:
            return head._logit_scale() * (F.normalize(q, dim=-1) @ F.normalize(codebook, dim=-1).t())
        return q @ codebook.t() / torch.clamp(head.log_tau.exp(), min=1e-3)

    def _per_user_losses(self, logits, targets):
        losses = []
        for attr in self.attr_order:
            if attr in self.ordinal_soft_attrs:
                target = build_ordinal_soft_labels(targets[attr], self.num_classes_map[attr],
                                                   self.ordinal_soft_sigma, logits[attr].device)
                losses.append(-(target * F.log_softmax(logits[attr].float(), -1)).sum(-1))
            else:
                losses.append(F.cross_entropy(logits[attr].float(), targets[attr], reduction='none'))
        return torch.stack(losses, -1)

class AmazonModel(_ProfileModel):
    gate_uncertainty_metric = "sqrt_gini"
    gate_uncertainty_formula = "1-sqrt(1-(1-sum(p_final**2))/(1-1/K))"

    def __init__(self, *args, evidence_dim=128, rho_min=.05, rho_max=.5,
                 initial_loss_beta=.2, **kwargs):
        self._initialize_gate(*args, gate_metric='excess_ce', **kwargs)
        if evidence_dim <= 0 or not 0 <= rho_min <= rho_max or not 0 <= initial_loss_beta <= 1:
            raise ValueError('invalid evidence_dim, rho range, or initial_loss_beta')
        self.evidence_dim = int(evidence_dim)
        self.rho_min, self.rho_max = float(rho_min), float(rho_max)
        self.initial_loss_beta = float(initial_loss_beta)
        self.evidence_memory_norm = nn.LayerNorm(self.hidden_size)
        self.evidence_memory_proj = nn.Linear(self.hidden_size, evidence_dim)
        self.evidence_query_norm = nn.LayerNorm(self.hidden_size)
        self.evidence_query_proj = nn.Linear(self.hidden_size, evidence_dim, bias=False)
        self.evidence_key_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_value_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_output_projs = nn.ModuleDict({
            attr: nn.Linear(evidence_dim, self.hidden_size) for attr in self.attr_order
        })
        # q1 == q0 at initialization; upstream evidence layers receive signal after this moves.
        for output in self.evidence_output_projs.values():
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, input_ids, attention_mask, age=None, gender=None, education=None,
                income=None, item_end_positions=None, item_mask=None, debug_printed=False,
                gate_epoch=None):
        if item_end_positions is None or item_mask is None:
            raise ValueError('margin-evidence forward requires item_end_positions and item_mask')
        targets = dict(age=age, gender=gender, education=education, income=income)
        provided = [x is not None for x in targets.values()]
        if any(provided) and not all(provided):
            raise ValueError('Supply all four attribute targets, or none')
        if self.training and any(provided) and gate_epoch is not None:
            epoch = gate_epoch.reshape(-1)
            if not epoch.numel() or not torch.all(epoch == epoch[0]):
                raise ValueError('All users must have the same nonempty training epoch')
            self.loss_gate_active.fill_(int(epoch[0]) >= self.gate_warmup_epochs)

        hidden = self.encode_backbone(input_ids, attention_mask)
        pooled = hidden[torch.arange(input_ids.size(0), device=input_ids.device), attention_mask.sum(-1) - 1]
        memory = self._item_memory(hidden, item_end_positions, item_mask)
        branches = {attr: pooled.clone() for attr in self.attr_order}
        q0 = torch.stack([getattr(self, f'{attr}_head').query_proj(branches[attr])
                          for attr in self.attr_order], dim=1)
        logits0 = {attr: self._retrieve(attr, q0[:, i]) for i, attr in enumerate(self.attr_order)}
        uncertainty, _, rho = self.margin_controls(logits0)
        context = self._attention(q0, memory, item_mask)
        q1 = self._final_queries(q0, context, rho, item_mask)
        logits1 = {attr: self._retrieve(attr, q1[:, i]) for i, attr in enumerate(self.attr_order)}
        # Register hooks after q1 scoring: hooks run during backward, so no
        # second backbone forward or recomputation of the query is needed.
        gini = self.gini_uncertainty(logits1)
        with torch.no_grad():
            gates = self.gate_min + (1 - self.gate_min) * (1 - gini) / (
                1 + gini / self.gate_loss_scale
            )
            if not bool(self.loss_gate_active):
                gates = torch.ones_like(gates)
        if self.training and any(provided):
            for i, attr in enumerate(self.attr_order):
                if branches[attr].requires_grad:
                    scale = gates[:, i].detach().to(branches[attr].dtype).unsqueeze(-1)
                    branches[attr].register_hook(lambda grad, scale=scale: grad * scale)
        result = dict(pooled=pooled, initial_logits=logits0, final_queries=q1,
                      margin_uncertainty=uncertainty, uncertainty_gates=gates, rho=rho,
                      gini_uncertainty=gini,
                      **{f'{attr}_logits': logits1[attr] for attr in self.attr_order})
        if not any(provided):
            result['loss'] = None
            return result
        loss0, loss1 = self._per_user_losses(logits0, targets), self._per_user_losses(logits1, targets)
        mixed = (1 - self.initial_loss_beta) * loss1 + self.initial_loss_beta * loss0
        per_attr = {attr: mixed[:, i].mean() for i, attr in enumerate(self.attr_order)}
        loss, weights = self.loss_weighting([per_attr[attr] for attr in self.attr_order])
        diagnostics = {f'loss_{attr}': per_attr[attr].detach() for attr in self.attr_order}
        diagnostics.update(loss_supervised=loss.detach(), metric_loss_gate_active=self.loss_gate_active.float().clone())
        for i, attr in enumerate(self.attr_order):
            q0_norm = q0[:, i].float().norm(dim=-1)
            correction = (q1[:, i] - q0[:, i].float()).norm(dim=-1)
            diagnostics.update({f'metric_initial_ce_{attr}': loss0[:, i].mean().detach(),
                                f'metric_final_ce_{attr}': loss1[:, i].mean().detach(),
                                f'metric_mixed_ce_{attr}': mixed[:, i].mean().detach(),
                                f'metric_margin_{attr}': uncertainty[:, i].mean(),
                                f'metric_gini_{attr}': gini[:, i].mean(),
                                f'metric_backbone_gate_{attr}': gates[:, i].mean(),
                                f'metric_rho_{attr}': rho[:, i].mean(),
                                f'metric_q0_norm_{attr}': q0_norm.mean().detach(),
                                f'metric_correction_to_q0_{attr}': (
                                    correction / q0_norm.clamp_min(1e-12)
                                ).mean().detach()})
        diagnostics.update({f'w_{key}': value for key, value in weights.items()})
        result.update(loss=loss, loss_dict=diagnostics, per_attr_loss=per_attr)
        return result

    def _initialize_gate(self, *args, gate_min=0.1, gate_loss_scale=0.5,
                 gate_warmup_epochs=1, gate_metric='excess_ce', **kwargs):
        if not math.isfinite(gate_min) or not 0 <= gate_min <= 1:
            raise ValueError('gate_min must be finite and in [0, 1]')
        if not math.isfinite(gate_loss_scale) or gate_loss_scale <= 0:
            raise ValueError('gate_loss_scale must be finite and positive')
        if (not math.isfinite(gate_warmup_epochs) or gate_warmup_epochs < 0
                or int(gate_warmup_epochs) != gate_warmup_epochs):
            raise ValueError('gate_warmup_epochs must be a nonnegative integer')
        if gate_metric not in ('ce', 'excess_ce'):
            raise ValueError('gate_metric must be ce or excess_ce')
        if kwargs.setdefault('loss_mode', 'uncertainty') not in ('fixed', 'uncertainty'):
            raise ValueError('loss_mode must be fixed or uncertainty')
        self._initialize_direct_queries(*args, **kwargs)
        self.gate_min = float(gate_min)
        self.gate_loss_scale = float(gate_loss_scale)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_metric = gate_metric
        self.register_buffer('loss_gate_active', torch.tensor(gate_warmup_epochs == 0))

    @torch.no_grad()
    def margin_controls(self, logits):
        margin, _, rho = self._base_margin_controls_evidence(logits)
        gini = self.gini_uncertainty(logits)
        gates = self.gate_min + (1 - self.gate_min) * (1 - gini) / (
            1 + gini / self.gate_loss_scale
        )
        if not bool(self.loss_gate_active):
            gates = torch.ones_like(gates)
        return margin, gates, rho

    def _base_set_mixed_precision_heads(self, device, backbone_dtype=torch.bfloat16):
        """Set large modules to BF16 while keeping small sensitive parameters FP32.

        Dtype policy:
          - backbone and four ResBlock query projections: backbone_dtype;
          - codebook embeddings, temperature scalars and loss weighting: FP32.

        Moving the whole model with ``dtype=...`` after this call would override the
        policy, so callers should use this method as their final dtype/device setup.
        """
        # Move every submodule to the target device without changing its current dtype.
        self.to(device=device)
        self.backbone.to(device=device, dtype=backbone_dtype)

        for attr in self.attr_order:
            head = getattr(self, f"{attr}_head")
            head.query_proj.to(device=device, dtype=backbone_dtype)
            head.codebook.to(device=device, dtype=torch.float32)

            # Temperature can be a Parameter (learnable) or a registered buffer.
            for scalar_name in ("log_logit_scale", "log_tau"):
                if not hasattr(head, scalar_name):
                    continue
                scalar = getattr(head, scalar_name)
                if isinstance(scalar, nn.Parameter):
                    scalar.data = scalar.data.to(device=device, dtype=torch.float32)
                    if scalar.grad is not None:
                        scalar.grad.data = scalar.grad.data.to(
                            device=device, dtype=torch.float32
                        )
                else:
                    setattr(
                        head,
                        scalar_name,
                        scalar.to(device=device, dtype=torch.float32),
                    )

        self.loss_weighting.to(device=device, dtype=torch.float32)
        return self

    def _base_dtype_info_heads(self):
        """Return the effective dtype policy for diagnostics and W&B config."""
        info = {
            "backbone": str(next(self.backbone.parameters()).dtype),
            "loss_weighting": str(next(self.loss_weighting.parameters(),
                                       next(self.loss_weighting.buffers(), None)).dtype),
        }
        for attr in self.attr_order:
            head = getattr(self, f"{attr}_head")
            info[f"{attr}_query_proj"] = str(next(head.query_proj.parameters()).dtype)
            info[f"{attr}_codebook"] = str(head.codebook.weight.dtype)
            scalar_name = "log_logit_scale" if head.use_cosine else "log_tau"
            info[f"{attr}_temperature"] = str(getattr(head, scalar_name).dtype)
        return info

    def _initialize_direct_queries(self, *args, **kwargs):
        self._initialize_heads(*args, **kwargs)
        for attr in self.attr_order:
            head = getattr(self, f"{attr}_head")
            head.query_proj = DirectQueryBlock(self.hidden_size)

    @torch.no_grad()
    def _base_margin_controls_evidence(self, logits):
        uncertainties = []
        for attr in self.attr_order:
            probabilities = F.softmax(logits[attr].float(), dim=-1)
            top2 = probabilities.topk(2, dim=-1).values
            uncertainties.append((1 - (top2[:, 0] - top2[:, 1])).clamp(0, 1))
        uncertainty = torch.stack(uncertainties, dim=-1)
        gates = self.gate_min + (1 - self.gate_min) * (1 - uncertainty) / (
            1 + uncertainty / self.gate_loss_scale
        )
        if not bool(self.loss_gate_active):
            gates = torch.ones_like(gates)
        rho = self.rho_min + (self.rho_max - self.rho_min) * uncertainty
        return uncertainty, gates, rho

    def _initialize_heads(
        self,
        peft_causal_lm,
        hidden_size,
        num_gender_classes,
        num_education_classes,
        num_age_classes,
        num_income_classes,
        use_cosine=True,
        temperature=0.07,
        learnable_temperature=True,
        loss_mode="uncertainty",           # "uncertainty" | "fixed"
        fixed_weights=None,


        ordinal_soft_attrs=("age",),
        ordinal_soft_sigma=1.0,
    ):
        nn.Module.__init__(self)
        self.backbone = peft_causal_lm
        self.hidden_size = hidden_size

        self.attr_order = ["gender", "education", "age", "income"]


        allowed_ordinal = {"age", "education", "income"}
        self.ordinal_soft_attrs = set(a for a in (ordinal_soft_attrs or []) if a in allowed_ordinal)
        self.ordinal_soft_sigma = ordinal_soft_sigma
        self.num_classes_map = {
            "gender": num_gender_classes,
            "education": num_education_classes,
            "age": num_age_classes,
            "income": num_income_classes,
        }


        head_kwargs = dict(
            hidden_size=hidden_size,
            use_cosine=use_cosine,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )
        self.gender_head = AttrRetrievalHead(num_classes=num_gender_classes, **head_kwargs)
        self.education_head = AttrRetrievalHead(num_classes=num_education_classes, **head_kwargs)
        self.age_head = AttrRetrievalHead(num_classes=num_age_classes, **head_kwargs)
        self.income_head = AttrRetrievalHead(num_classes=num_income_classes, **head_kwargs)


        self.loss_weighting = MultiTaskLossWeighting(
            num_tasks=4,
            mode=loss_mode,
            fixed_weights=fixed_weights,
        )

class ML1MModel(_ProfileModel):
    gate_uncertainty_metric = "sqrt_gini"
    gate_uncertainty_formula = "1-sqrt(1-(1-sum(p_final**2))/(1-1/K))"

    def __init__(self, *args, evidence_dim=128, rho_min=.05, rho_max=.5,
                 initial_loss_beta=.2, **kwargs):
        self._initialize_gate(*args, **kwargs)
        if evidence_dim <= 0 or not 0 <= rho_min <= rho_max or not 0 <= initial_loss_beta <= 1:
            raise ValueError('invalid evidence_dim, rho range, or initial_loss_beta')
        self.evidence_dim = int(evidence_dim)
        self.rho_min, self.rho_max = float(rho_min), float(rho_max)
        self.initial_loss_beta = float(initial_loss_beta)
        self.evidence_memory_norm = nn.LayerNorm(self.hidden_size)
        self.evidence_memory_proj = nn.Linear(self.hidden_size, evidence_dim)
        self.evidence_query_norm = nn.LayerNorm(self.hidden_size)
        self.evidence_query_proj = nn.Linear(self.hidden_size, evidence_dim, bias=False)
        self.evidence_key_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_value_proj = nn.Linear(evidence_dim, evidence_dim, bias=False)
        self.evidence_output_projs = nn.ModuleDict({
            attr: nn.Linear(evidence_dim, self.hidden_size) for attr in self.attr_order
        })
        # q1 == q0 at initialization; upstream evidence layers receive signal after this moves.
        for output in self.evidence_output_projs.values():
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, input_ids, attention_mask, age=None, gender=None, occupation=None, item_end_positions=None, item_mask=None, debug_printed=False,
                gate_epoch=None):
        if item_end_positions is None or item_mask is None:
            raise ValueError('margin-evidence forward requires item_end_positions and item_mask')
        targets = dict(age=age, gender=gender, occupation=occupation)
        provided = [x is not None for x in targets.values()]
        if any(provided) and not all(provided):
            raise ValueError('Supply all three attribute targets, or none')
        if self.training and any(provided) and gate_epoch is not None:
            epoch = gate_epoch.reshape(-1)
            if not epoch.numel() or not torch.all(epoch == epoch[0]):
                raise ValueError('All users must have the same nonempty training epoch')
            self.loss_gate_active.fill_(int(epoch[0]) >= self.gate_warmup_epochs)

        hidden = self.encode_backbone(input_ids, attention_mask)
        pooled = hidden[torch.arange(input_ids.size(0), device=input_ids.device), attention_mask.sum(-1) - 1]
        memory = self._item_memory(hidden, item_end_positions, item_mask)
        branches = {attr: pooled.clone() for attr in self.attr_order}
        q0 = torch.stack([getattr(self, f'{attr}_head').query_proj(branches[attr])
                          for attr in self.attr_order], dim=1)
        logits0 = {attr: self._retrieve(attr, q0[:, i]) for i, attr in enumerate(self.attr_order)}
        uncertainty, _, rho = self.margin_controls(logits0)
        context = self._attention(q0, memory, item_mask)
        q1 = self._final_queries(q0, context, rho, item_mask)
        logits1 = {attr: self._retrieve(attr, q1[:, i]) for i, attr in enumerate(self.attr_order)}
        # Register hooks after q1 scoring: hooks run during backward, so no
        # second backbone forward or recomputation of the query is needed.
        gini = self.gini_uncertainty(logits1)
        with torch.no_grad():
            gates = self.gate_min + (1 - self.gate_min) * (1 - gini) / (
                1 + gini / self.gate_loss_scale
            )
            if not bool(self.loss_gate_active):
                gates = torch.ones_like(gates)
        if self.training and any(provided):
            for i, attr in enumerate(self.attr_order):
                if branches[attr].requires_grad:
                    scale = gates[:, i].detach().to(branches[attr].dtype).unsqueeze(-1)
                    branches[attr].register_hook(lambda grad, scale=scale: grad * scale)
        result = dict(pooled=pooled, initial_logits=logits0, final_queries=q1,
                      margin_uncertainty=uncertainty, uncertainty_gates=gates, rho=rho,
                      gini_uncertainty=gini,
                      **{f'{attr}_logits': logits1[attr] for attr in self.attr_order})
        if not any(provided):
            result['loss'] = None
            return result
        loss0, loss1 = self._per_user_losses(logits0, targets), self._per_user_losses(logits1, targets)
        mixed = (1 - self.initial_loss_beta) * loss1 + self.initial_loss_beta * loss0
        per_attr = {attr: mixed[:, i].mean() for i, attr in enumerate(self.attr_order)}
        loss, weights = self.loss_weighting([per_attr[attr] for attr in self.attr_order])
        diagnostics = {f'loss_{attr}': per_attr[attr].detach() for attr in self.attr_order}
        diagnostics.update(loss_supervised=loss.detach(), metric_loss_gate_active=self.loss_gate_active.float().clone())
        for i, attr in enumerate(self.attr_order):
            q0_norm = q0[:, i].float().norm(dim=-1)
            correction = (q1[:, i] - q0[:, i].float()).norm(dim=-1)
            diagnostics.update({f'metric_initial_ce_{attr}': loss0[:, i].mean().detach(),
                                f'metric_final_ce_{attr}': loss1[:, i].mean().detach(),
                                f'metric_mixed_ce_{attr}': mixed[:, i].mean().detach(),
                                f'metric_margin_{attr}': uncertainty[:, i].mean(),
                                f'metric_gini_{attr}': gini[:, i].mean(),
                                f'metric_backbone_gate_{attr}': gates[:, i].mean(),
                                f'metric_rho_{attr}': rho[:, i].mean(),
                                f'metric_q0_norm_{attr}': q0_norm.mean().detach(),
                                f'metric_correction_to_q0_{attr}': (
                                    correction / q0_norm.clamp_min(1e-12)
                                ).mean().detach()})
        diagnostics.update({f'w_{key}': value for key, value in weights.items()})
        result.update(loss=loss, loss_dict=diagnostics, per_attr_loss=per_attr)
        return result

    def _initialize_gate(self, *args, gate_min=.1, gate_loss_scale=.5, gate_warmup_epochs=1, **kwargs):
        self._initialize_heads(*args, **kwargs)
        if self.query_head_type != 'direct':
            raise ValueError('The final model requires direct query heads')
        if not math.isfinite(gate_min) or not 0 <= gate_min <= 1:
            raise ValueError('gate_min must be finite and in [0, 1]')
        if not math.isfinite(gate_loss_scale) or gate_loss_scale <= 0:
            raise ValueError('gate_loss_scale must be finite and positive')
        if gate_warmup_epochs < 0 or int(gate_warmup_epochs) != gate_warmup_epochs:
            raise ValueError('gate_warmup_epochs must be a nonnegative integer')
        if any(n < 2 for n in self.num_classes_map.values()):
            raise ValueError('Margin requires at least two classes per attribute')
        self.gate_min = float(gate_min)
        self.gate_loss_scale = float(gate_loss_scale)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_metric = 'margin'
        self.register_buffer('loss_gate_active', torch.tensor(gate_warmup_epochs == 0))

    @torch.no_grad()
    def margin_controls(self, logits):
        uncertainties = []
        for attr in self.attr_order:
            probabilities = F.softmax(logits[attr].float(), dim=-1)
            top2 = probabilities.topk(2, dim=-1).values
            uncertainties.append((1 - (top2[:, 0] - top2[:, 1])).clamp(0, 1))
        uncertainty = torch.stack(uncertainties, dim=-1)
        gates = self.gate_min + (1 - self.gate_min) * (1 - uncertainty) / (
            1 + uncertainty / self.gate_loss_scale
        )
        if not bool(self.loss_gate_active):
            gates = torch.ones_like(gates)
        rho = self.rho_min + (self.rho_max - self.rho_min) * uncertainty
        return uncertainty, gates, rho

    def _base_set_mixed_precision_heads(self, device, backbone_dtype=torch.bfloat16):
        self.to(device=device)
        self.backbone.to(device=device, dtype=backbone_dtype)
        for attr in self.attr_order:
            head = getattr(self, f"{attr}_head")
            head.query_proj.to(device=device, dtype=backbone_dtype)
            head.codebook.to(device=device, dtype=torch.float32)
            for scalar_name in ("log_logit_scale", "log_tau"):
                if not hasattr(head, scalar_name):
                    continue
                scalar = getattr(head, scalar_name)
                if isinstance(scalar, nn.Parameter):
                    scalar.data = scalar.data.to(device=device, dtype=torch.float32)
                else:
                    setattr(
                        head,
                        scalar_name,
                        scalar.to(device=device, dtype=torch.float32),
                    )
        self.loss_weighting.to(device=device, dtype=torch.float32)
        return self

    def _base_dtype_info_heads(self):
        first_weight = next(
            self.loss_weighting.parameters(),
            next(self.loss_weighting.buffers(), None),
        )
        info = {
            "backbone": str(next(self.backbone.parameters()).dtype),
            "loss_weighting": str(first_weight.dtype),
        }
        for attr in self.attr_order:
            head = getattr(self, f"{attr}_head")
            info[f"{attr}_query_proj"] = str(next(head.query_proj.parameters()).dtype)
            info[f"{attr}_codebook"] = str(head.codebook.weight.dtype)
        return info

    def _initialize_heads(
        self,
        peft_causal_lm,
        hidden_size,
        num_gender_classes,
        num_occupation_classes,
        num_age_classes,
        query_head_type="direct",
        use_cosine=True,
        temperature=0.07,
        learnable_temperature=True,
        loss_mode="uncertainty",
        fixed_weights=None,
        ordinal_soft_attrs=("age",),
        ordinal_soft_sigma=1.0,
    ):
        nn.Module.__init__(self)
        if query_head_type not in {"direct", "residual"}:
            raise ValueError(f"unsupported query_head_type: {query_head_type}")

        self.backbone = peft_causal_lm
        self.hidden_size = hidden_size
        self.attr_order = list(MOVIELENS_ATTRS)
        self.query_head_type = query_head_type
        self.num_classes_map = {
            "gender": num_gender_classes,
            "occupation": num_occupation_classes,
            "age": num_age_classes,
        }
        allowed_ordinal = {"age"}
        self.ordinal_soft_attrs = {
            attr for attr in (ordinal_soft_attrs or ()) if attr in allowed_ordinal
        }
        self.ordinal_soft_sigma = ordinal_soft_sigma

        head_kwargs = {
            "hidden_size": hidden_size,
            "use_cosine": use_cosine,
            "temperature": temperature,
            "learnable_temperature": learnable_temperature,
        }
        for attr in self.attr_order:
            head = AttrRetrievalHead(
                num_classes=self.num_classes_map[attr], **head_kwargs
            )
            if query_head_type == "direct":
                head.query_proj = DirectQueryBlock(hidden_size)
            setattr(self, f"{attr}_head", head)

        self.loss_weighting = MultiTaskLossWeighting(
            num_tasks=len(self.attr_order),
            mode=loss_mode,
            fixed_weights=fixed_weights,
        )
