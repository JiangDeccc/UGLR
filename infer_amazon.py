"""Restore and evaluate the final Amazon model."""
from model import AmazonModel, build_ordinal_soft_labels, soft_cross_entropy
from train_amazon import EVIDENCE_KEYS, MarginEvidenceDataset, VARIANT
import os
import json
import pickle
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from rpg_metrics import macro_f1_score
from rpg_history_encoding import encode_recent_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--test_file", type=str,
                        default="./data/amazon/test_user_history_his200.pkl")
    parser.add_argument("--label_file", type=str,
                        default="./data/amazon/true_user_profile.pkl")
    parser.add_argument("--lora_dir", type=str,
                        default="./lora/rpg_codebook/best_lora_adapter")
    parser.add_argument("--ckpt_path", type=str,
                        default="./lora/rpg_codebook/best_checkpoint.pt")
    parser.add_argument("--output_path", type=str,
                        default="./lora/rpg_codebook/test_distribution.pkl")
    parser.add_argument("--max_his_len", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--filter_unknown", type=int, default=0,
                        help='1=exclude users with unknown labels during evaluation, as in training')
    parser.add_argument("--save_jsonl", type=int, default=0)
    return parser.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def move_batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


UNKNOWN_LABEL_MAP = {"gender": 3, "education": 3, "income": 6}


def build_model(args, tokenizer):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    cfg = ckpt["config"]

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    base_model = PeftModel.from_pretrained(base_model, args.lora_dir, is_trainable=False)

    model = AmazonModel(
        **constructor_options(cfg),
        peft_causal_lm=base_model,
        hidden_size=cfg["hidden_size"],
        num_gender_classes=cfg["num_gender_classes"],
        num_education_classes=cfg["num_education_classes"],
        num_age_classes=cfg["num_age_classes"],
        num_income_classes=cfg["num_income_classes"],
        use_cosine=cfg["use_cosine"],
        temperature=cfg["temperature"],
        learnable_temperature=cfg["learnable_temperature"],
        loss_mode=cfg["loss_mode"],
        fixed_weights=cfg.get("fixed_weights"),
        ordinal_soft_attrs=cfg.get("ordinal_soft_attrs", ("age",)),
        ordinal_soft_sigma=cfg.get("ordinal_soft_sigma", 1.0),
    )

    missing, unexpected = model.load_state_dict(ckpt["other_state_dict"], strict=False)
    real_missing = [k for k in missing if not k.startswith("backbone.")]
    if real_missing or unexpected:
        raise ValueError(f"Incomplete final-model checkpoint: {real_missing=}, {unexpected=}")
    print("Missing keys:", real_missing)
    print("Unexpected keys:", unexpected)
    return model


def compute_known_loss(model, logits_map, batch, known_mask):
    'Compute checkpoint-configured loss only for users with four known labels.'
    known_count = int(known_mask.sum().item())
    if known_count == 0:
        return None, 0

    per_attr_loss = {}
    initial_logits = logits_map.get("initial_logits")
    initial_loss_beta = getattr(model, "initial_loss_beta", None)
    if initial_logits is not None and initial_loss_beta is None:
        raise ValueError("Initial logits require an initial_loss_beta on the model")
    for attr in model.attr_order:
        logits = logits_map[attr][known_mask]
        targets = batch[attr][known_mask]
        if attr in model.ordinal_soft_attrs:
            soft_targets = build_ordinal_soft_labels(
                targets,
                model.num_classes_map[attr],
                sigma=model.ordinal_soft_sigma,
                device=logits.device,
            )
            final_loss = soft_cross_entropy(logits, soft_targets)
            if initial_logits is not None:
                initial_loss = soft_cross_entropy(initial_logits[attr][known_mask], soft_targets)
        else:
            final_loss = F.cross_entropy(logits, targets)
            if initial_logits is not None:
                initial_loss = F.cross_entropy(initial_logits[attr][known_mask], targets)
        per_attr_loss[attr] = (
            (1 - initial_loss_beta) * final_loss + initial_loss_beta * initial_loss
            if initial_logits is not None else final_loss
        )

    total_loss, _ = model.loss_weighting(
        [per_attr_loss[attr] for attr in model.attr_order]
    )
    return total_loss, known_count


def to_jsonable(value):
    """Convert DataLoader-collated scalars/arrays to JSON-serializable values."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@torch.no_grad()
def infer_and_save(model, loader, device, output_path, save_jsonl=False):
    model.eval()
    results = []
    correct = {"age": 0, "gender": 0, "education": 0, "income": 0}
    all_targets = {attr: [] for attr in correct}
    all_predictions = {attr: [] for attr in correct}
    unknown_labels = {attr: 0 for attr in UNKNOWN_LABEL_MAP}
    unknown_users = 0
    total_loss = 0.0
    loss_count = 0
    total = 0

    attrs = ["age", "gender", "education", "income"]

    for batch in tqdm(loader, desc="Infer test"):
        batch = move_batch_to_device(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **{key: batch[key] for key in ("item_end_positions", "item_mask")
               if key in batch},
        )
        logits = {a: outputs[f"{a}_logits"].float() for a in attrs}
        probs = {a: torch.softmax(logits[a], dim=-1) for a in attrs}
        preds = {a: logits[a].argmax(dim=-1) for a in attrs}

        bs = batch["input_ids"].size(0)
        total += bs

        known_mask = torch.ones(bs, dtype=torch.bool, device=device)
        for attr, unknown_id in UNKNOWN_LABEL_MAP.items():
            attr_unknown = batch[attr].eq(unknown_id)
            unknown_labels[attr] += int(attr_unknown.sum().item())
            known_mask &= ~attr_unknown
        unknown_users += int((~known_mask).sum().item())

        for a in attrs:
            correct[a] += (preds[a] == batch[a]).sum().item()
            all_predictions[a].extend(preds[a].detach().cpu().tolist())
            all_targets[a].extend(batch[a].detach().cpu().tolist())

        loss_logits = {**logits, "initial_logits": outputs["initial_logits"]} \
            if "initial_logits" in outputs else logits
        loss, batch_loss_count = compute_known_loss(model, loss_logits, batch, known_mask)
        if loss is not None:
            total_loss += float(loss.item()) * batch_loss_count
            loss_count += batch_loss_count

        for i in range(bs):
            row = {
                "sample_key": to_jsonable(batch["sample_key"][i]),
                "prompt": batch["prompt"][i],
            }
            for a in attrs:
                row[f"true_{a}"] = int(batch[a][i].item())
                row[f"pred_{a}"] = int(preds[a][i].item())
                row[f"{a}_logits"] = logits[a][i].cpu().tolist()
                row[f"{a}_probs"] = probs[a][i].cpu().tolist()
            results.append(row)

    metrics = {f"{a}_acc": correct[a] / total for a in attrs}
    metrics["avg_acc"] = sum(correct.values()) / (4 * total)
    for a in attrs:
        metrics[f"{a}_macro_f1"] = macro_f1_score(
            all_targets[a], all_predictions[a]
        )
    metrics["avg_macro_f1"] = sum(
        metrics[f"{a}_macro_f1"] for a in attrs
    ) / len(attrs)
    metrics.update({
        "loss": total_loss / loss_count if loss_count else None,
        "num_samples": total,
        "loss_count": loss_count,
        "unknown_users": unknown_users,
        "unknown_labels": unknown_labels,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"metrics": metrics, "results": results}, f)
    print("Saved to:", output_path)
    print("Metrics:", metrics)

    if save_jsonl:
        jsonl_path = output_path.rsplit(".", 1)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("Saved jsonl to:", jsonl_path)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    test_dataset = build_test_dataset(
        args.test_file, args.label_file, tokenizer,
        args.max_his_len, args.max_len, filter_unknown=bool(args.filter_unknown),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.set_mixed_precision(device=device, backbone_dtype=torch.bfloat16)
    print("Mixed-precision dtypes:", model.dtype_info())

    infer_and_save(model, test_loader, device, args.output_path,
                   save_jsonl=bool(args.save_jsonl))


def constructor_options(config):
    if config.get('model_variant') != VARIANT:
        raise ValueError(f"Expected {VARIANT!r}, got {config.get('model_variant')!r}")
    setup = config.get('model_setup', {})
    expected = {
        'semantic_init_version': 'last_text_global_center_norm_match_v1',
        'semantic_pooling': 'last_valid_text_token',
        'semantic_similarity_penalty': True,
        'semantic_norm_matching': 'per_attribute_preinit_mean_row_norm',
        'gate_uncertainty_metric': AmazonModel.gate_uncertainty_metric,
        'gate_uncertainty_formula': AmazonModel.gate_uncertainty_formula,
        'gate_uncertainty_normalized': True,
        'evidence_uncertainty_metric': 'margin',
        'gate_probability_source': 'train_mode_q1_logits',
    }
    for key, value in expected.items():
        if setup.get(key) != value:
            raise ValueError(f'Incompatible uncertainty metadata: {key}')
    missing = [key for key in EVIDENCE_KEYS if key not in setup]
    if missing:
        raise ValueError(f'Missing margin-evidence metadata: {missing}')
    if setup.get('memory_detach') != 'before_FP32_memory_norm_and_projection':
        raise ValueError('Unexpected or missing margin-evidence detach contract')
    return {key: setup[key] for key in EVIDENCE_KEYS}

def build_test_dataset(test_file_path, label_file_path, tokenizer, max_his_len, max_len,
                       filter_unknown=False):
    inputs, labels, keys = [], [], []
    label_file = load_pickle(label_file_path)
    for key, history in load_pickle(test_file_path).items():
        label = label_file[key]['label_id']
        if filter_unknown and any(label[a] == u for a, u in UNKNOWN_LABEL_MAP.items()):
            continue
        inputs.append(history); labels.append(label); keys.append(key)
    return MarginEvidenceDataset(inputs, labels, keys, tokenizer, max_his_len=max_his_len, max_len=max_len)

if __name__ == "__main__":
    main()
