from model import ML1MModel, MOVIELENS_ATTRS
from semantic_init import initialize_codebooks, validate_options
"""Train three real MovieLens targets: gender, occupation and age."""

import argparse
import math
import random
import json
import os
import pickle

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from rpg_metrics import macro_f1_score


VARIANT = "sqrt_gini_margin_evidence_semantic_spread_final_query_gate"


def model_options(args):
    keys = ("gate_min", "gate_loss_scale", "gate_warmup_epochs", "evidence_dim",
            "rho_min", "rho_max", "initial_loss_beta")
    return {key: getattr(args, key) for key in keys}


def model_inputs(batch):
    return {key: batch[key] for key in (
        "input_ids", "attention_mask", *MOVIELENS_ATTRS,
        "gate_epoch", "item_end_positions", "item_mask") if key in batch}


WEIGHT_DECAY = 1e-2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Three-head RPG codebook training for MovieLens"
    )
    parser.add_argument(
        "--model_path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--train_file",
        default="./data/ml1m/train_user_history_his200.pkl",
    )
    parser.add_argument(
        "--valid_file",
        default="./data/ml1m/val_user_history_his200.pkl",
    )
    parser.add_argument(
        "--test_file",
        default="./data/ml1m/test_user_history_his200.pkl",
    )
    parser.add_argument(
        "--label_file",
        default="./data/ml1m/true_user_profile.pkl",
    )
    parser.add_argument(
        "--lora_save_path", default="./lora/rpg_codebook_movielens_sqrt_gini_margin_evidence_semantic_spread_final_query_gate/"
    )
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=1e-5)
    parser.add_argument("--head_no_wd_lr", type=float, default=1e-4)
    parser.add_argument("--codebook_lr", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--early_stop_step", type=int, default=0)
    parser.add_argument("--max_his_len", "--recent_n", type=int, default=100)
    parser.add_argument("--max_len", "--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--query_head_type", choices=["direct"], default="direct"
    )
    parser.add_argument("--use_cosine", type=int, choices=[0, 1], default=1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--learnable_temperature", type=int, choices=[0, 1], default=1
    )
    parser.add_argument(
        "--loss_mode", choices=["uncertainty", "fixed"], default="uncertainty"
    )
    parser.add_argument("--fixed_weights", default="1,1,1")
    parser.add_argument("--ordinal_soft_attrs", default="")
    parser.add_argument("--ordinal_soft_sigma", type=float, default=0.5)
    parser.add_argument("--selection_metric", choices=["avg_acc", "avg_macro_f1"], default="avg_acc")
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--wandb_project", default="movielens_profile_rpg_codebook")
    parser.add_argument("--wandb_run_name", default="llama3_rpg_codebook_movielens_sqrt_gini_margin_evidence_semantic_spread_final_query_gate")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--gate_min", type=float, default=.1)
    parser.add_argument("--gate_loss_scale", type=float, default=.5)
    parser.add_argument("--gate_warmup_epochs", type=int, default=1)
    parser.add_argument("--evidence_dim", type=int, default=128)
    parser.add_argument("--evidence_lr", type=float, default=1e-4)
    parser.add_argument("--rho_min", type=float, default=.05)
    parser.add_argument("--rho_max", type=float, default=.5)
    parser.add_argument("--initial_loss_beta", type=float, default=.2)
    parser.add_argument('--semantic_center_strength', type=float, default=1.)
    parser.add_argument('--semantic_similarity_weight', type=float, default=.1)
    parser.add_argument('--semantic_similarity_steps', type=int, default=200)
    parser.add_argument('--semantic_similarity_lr', type=float, default=.05)
    parser.add_argument('--semantic_within_cap', type=float, default=.5)
    parser.add_argument('--semantic_cross_cap', type=float, default=.3)
    args = parser.parse_args(argv)
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.batch_size != 1:
        parser.error("Unpadded training requires --batch_size 1; use gradient accumulation")
    if min(args.epoch, args.accumulation_steps, args.max_his_len, args.max_len) <= 0:
        parser.error("epoch, accumulation_steps, history and token limits must be positive")
    if args.ordinal_soft_attrs.strip() not in ("", "age"):
        parser.error("Only age supports ordinal smoothing")
    if not math.isfinite(args.ordinal_soft_sigma) or args.ordinal_soft_sigma <= 0:
        parser.error("ordinal_soft_sigma must be finite and positive")
    if args.query_head_type != "direct":
        parser.error("The final model requires direct query heads")
    if not math.isfinite(args.gate_min) or not 0 <= args.gate_min <= 1:
        parser.error("gate_min must be in [0, 1]")
    if not math.isfinite(args.gate_loss_scale) or args.gate_loss_scale <= 0 or args.gate_warmup_epochs < 0:
        parser.error("gate_loss_scale must be positive and warmup epochs nonnegative")
    if not all(math.isfinite(v) for v in (args.rho_min, args.rho_max, args.initial_loss_beta, args.evidence_lr)):
        parser.error("Evidence parameters must be finite")
    if not 0 <= args.rho_min <= args.rho_max or not 0 <= args.initial_loss_beta <= 1 or args.evidence_dim <= 0 or args.evidence_lr <= 0:
        parser.error("Invalid evidence dimension, learning rate, rho range or initial loss weight")
    return args


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def normalize_label(profile):
    return {attr: int(profile["label_id"][attr]) for attr in MOVIELENS_ATTRS}


def load_split(history_path, profiles):
    histories = load_pickle(history_path)
    inputs, labels, keys = [], [], []
    for key, history in histories.items():
        if key not in profiles:
            raise KeyError(f"missing profile for user {key!r}")
        if not isinstance(history, list) or not history:
            raise TypeError(f"history for user {key!r} must be a non-empty list")
        inputs.append(history)
        labels.append(normalize_label(profiles[key]))
        keys.append(key)
    return inputs, labels, keys


def validate_labels(split_name, labels):
    if not labels:
        raise ValueError(f"{split_name} is empty")
    for attr in MOVIELENS_ATTRS:
        values = sorted({label[attr] for label in labels})
        expected = list(range(max(values) + 1))
        if values != expected:
            raise ValueError(
                f"{split_name} {attr} labels must be contiguous from 0: {values}"
            )


class MovieLensDataset(Dataset):
    def __init__(self, histories, labels, keys, tokenizer, max_his_len, max_len):
        self.epoch = 0
        self.histories = histories
        self.labels = labels
        self.keys = keys
        self.tokenizer = tokenizer
        self.max_his_len = max_his_len
        self.max_len = max_len

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        items = self.histories[index][-self.max_his_len :]
        history_text = "\n".join(items)
        prefix = (
            "Task: Infer age group, gender and occupation from movie rating history.\n\n"
            "Here is the user's interaction history:\n"
        )
        prompt = prefix + history_text + "\n"
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        item = {
            "sample_key": self.keys[index],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        item.update(
            {
                attr: torch.tensor(self.labels[index][attr], dtype=torch.long)
                for attr in MOVIELENS_ATTRS
            }
        )
        item["gate_epoch"] = torch.tensor(self.epoch, dtype=torch.long)
        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        visible_end = max((right for left, right in offsets if right > left), default=0)
        ends, visible, cursor = [], [], len(prefix)
        for history_item in items:
            end = cursor + len(history_item)
            tokens = [i for i, (left, right) in enumerate(offsets)
                      if right > left and left < end and right > cursor]
            complete = bool(tokens) and end <= visible_end
            ends.append(tokens[-1] if complete else 0)
            visible.append(complete)
            cursor = end + 1
        pad = self.max_his_len - len(ends)
        item["item_end_positions"] = torch.tensor(ends + [0] * pad, dtype=torch.long)
        item["item_mask"] = torch.tensor(visible + [False] * pad, dtype=torch.bool)
        return item


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device, description, output_path=None, save_jsonl=False):
    model.eval()
    records = []
    total = 0
    loss_sum = 0.0
    correct = {attr: 0 for attr in MOVIELENS_ATTRS}
    targets = {attr: [] for attr in MOVIELENS_ATTRS}
    predictions = {attr: [] for attr in MOVIELENS_ATTRS}
    tuple_correct = 0

    for batch in tqdm(loader, desc=description, leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(**model_inputs(batch))
        batch_size = batch["input_ids"].size(0)
        total += batch_size
        loss_sum += outputs["loss"].item() * batch_size

        if output_path:
            keys = batch["sample_key"]
            for i in range(batch_size):
                key = keys[i].item() if torch.is_tensor(keys[i]) else keys[i]
                records.append({"user_id": key,
                    "labels": {attr: int(batch[attr][i]) for attr in MOVIELENS_ATTRS},
                    "predictions": {attr: int(outputs[f"{attr}_logits"][i].argmax()) for attr in MOVIELENS_ATTRS},
                    "probabilities": {attr: outputs[f"{attr}_logits"][i].float().softmax(-1).cpu().tolist()
                                      for attr in MOVIELENS_ATTRS}})

        prediction_matrix = []
        target_matrix = []
        for attr in MOVIELENS_ATTRS:
            predicted = outputs[f"{attr}_logits"].argmax(dim=-1)
            correct[attr] += (predicted == batch[attr]).sum().item()
            predictions[attr].extend(predicted.cpu().tolist())
            targets[attr].extend(batch[attr].cpu().tolist())
            prediction_matrix.append(predicted)
            target_matrix.append(batch[attr])
        tuple_correct += (
            torch.stack(prediction_matrix, dim=1)
            .eq(torch.stack(target_matrix, dim=1))
            .all(dim=1)
            .sum()
            .item()
        )

    metrics = {"loss": loss_sum / total}
    metrics.update({f"{attr}_acc": correct[attr] / total for attr in MOVIELENS_ATTRS})
    metrics["avg_acc"] = sum(correct.values()) / (len(MOVIELENS_ATTRS) * total)
    for attr in MOVIELENS_ATTRS:
        metrics[f"{attr}_macro_f1"] = macro_f1_score(
            targets[attr], predictions[attr]
        )
    metrics["avg_macro_f1"] = sum(
        metrics[f"{attr}_macro_f1"] for attr in MOVIELENS_ATTRS
    ) / len(MOVIELENS_ATTRS)
    metrics["tuple_acc"] = tuple_correct / total
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump({"metrics": metrics, "predictions": records}, handle)
        if save_jsonl:
            with open(os.path.splitext(output_path)[0] + ".jsonl", "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def non_backbone_state_dict(model):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    profiles = load_pickle(args.label_file)
    train_x, train_y, train_keys = load_split(args.train_file, profiles)
    val_x, val_y, val_keys = load_split(args.valid_file, profiles)
    test_x, test_y, test_keys = load_split(args.test_file, profiles)
    validate_labels("train", train_y)
    print(f"train={len(train_x)}, val={len(val_x)}, test={len(test_x)}")

    num_classes = {
        attr: max(label[attr] for label in train_y) + 1 for attr in MOVIELENS_ATTRS
    }
    print("num_classes:", num_classes)

    for split_name, labels in (("val", val_y), ("test", test_y)):
        for label in labels:
            for attr in MOVIELENS_ATTRS:
                if not 0 <= label[attr] < num_classes[attr]:
                    raise ValueError(f"{split_name}: out-of-range {attr} label {label[attr]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def make_loader(histories, labels, keys, shuffle):
        dataset = MovieLensDataset(
            histories, labels, keys, tokenizer, args.max_his_len, args.max_len
        )
        generator = torch.Generator().manual_seed(args.seed) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=args.num_workers,
        )

    train_loader = make_loader(train_x, train_y, train_keys, True)
    val_loader = make_loader(val_x, val_y, val_keys, False)
    test_loader = make_loader(test_x, test_y, test_keys, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_kwargs = {
        "device_map": None,
        "torch_dtype": backbone_dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    fixed_weights = [float(value) for value in args.fixed_weights.split(",")]
    if len(fixed_weights) != len(MOVIELENS_ATTRS):
        raise ValueError("fixed_weights must contain three comma-separated values")
    ordinal_soft_attrs = [
        value.strip() for value in args.ordinal_soft_attrs.split(",") if value.strip()
    ]
    invalid_ordinal = set(ordinal_soft_attrs) - {"age"}
    if invalid_ordinal:
        raise ValueError(f"unsupported ordinal_soft_attrs: {sorted(invalid_ordinal)}")

    model = ML1MModel(
        peft_causal_lm=base_model,
        hidden_size=base_model.config.hidden_size,
        num_gender_classes=num_classes["gender"],
        num_occupation_classes=num_classes["occupation"],
        num_age_classes=num_classes["age"],
        query_head_type=args.query_head_type,
        use_cosine=bool(args.use_cosine),
        temperature=args.temperature,
        learnable_temperature=bool(args.learnable_temperature),
        loss_mode=args.loss_mode,
        fixed_weights=fixed_weights,
        ordinal_soft_attrs=ordinal_soft_attrs,
        ordinal_soft_sigma=args.ordinal_soft_sigma,
        **model_options(args),
    ).set_mixed_precision(device, backbone_dtype)
    model_setup = initialize_codebooks(model, tokenizer, device, args, spread=True, dataset="ml1m")
    model_setup.update(gate_uncertainty_metric=model.gate_uncertainty_metric,
        gate_uncertainty_formula=model.gate_uncertainty_formula,
        gate_probability_source='train_mode_q1_logits', evidence_uncertainty_metric='margin',
        memory_detach='before_FP32_memory_norm_and_projection', **model_options(args))
    print("Mixed-precision dtypes:", model.dtype_info())

    parameter_groups = {"lora": [], "head_wd": [], "head_no_wd": [], "codebook": [], "evidence": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            parameter_groups["lora"].append(parameter)
        elif name.startswith("evidence_"):
            parameter_groups["evidence"].append(parameter)
        elif "codebook" in name:
            parameter_groups["codebook"].append(parameter)
        elif any(token in name for token in ("log_logit_scale", "log_tau", "log_var")):
            parameter_groups["head_no_wd"].append(parameter)
        else:
            parameter_groups["head_wd"].append(parameter)

    optimizer = torch.optim.AdamW([
        {"name": "lora", "params": parameter_groups["lora"], "lr": args.lr,
         "weight_decay": WEIGHT_DECAY},
        {"name": "head_wd", "params": parameter_groups["head_wd"], "lr": args.head_lr,
         "weight_decay": WEIGHT_DECAY},
        {"name": "head_no_wd", "params": parameter_groups["head_no_wd"],
         "lr": args.head_no_wd_lr, "weight_decay": 0.0},
        {"name": "codebook", "params": parameter_groups["codebook"],
         "lr": args.codebook_lr, "weight_decay": 0.0},
        {"name": "evidence", "params": parameter_groups["evidence"],
         "lr": args.evidence_lr, "weight_decay": WEIGHT_DECAY},
    ])
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    total_updates = max(updates_per_epoch * args.epoch, 1)
    warmup_updates = max(total_updates // 10, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )

    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "num_classes": num_classes, "model_setup": model_setup},
        )

    os.makedirs(args.lora_save_path, exist_ok=True)

    def save_checkpoint(tag, epoch=None, metrics=None):
        adapter_dir = os.path.join(args.lora_save_path, f"{tag}_lora_adapter")
        model.backbone.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        config = {
            "dataset": "ml-1m",
            "model_variant": VARIANT,
            "model_setup": model_setup,
            "model_options": model_options(args),
            "model_path": args.model_path,
            "max_his_len": args.max_his_len,
            "max_len": args.max_len,
            "attributes": list(MOVIELENS_ATTRS),
            "num_classes": num_classes,
            "hidden_size": model.hidden_size,
            "query_head_type": args.query_head_type,
            "use_cosine": bool(args.use_cosine),
            "temperature": args.temperature,
            "learnable_temperature": bool(args.learnable_temperature),
            "loss_mode": args.loss_mode,
            "fixed_weights": fixed_weights,
            "ordinal_soft_attrs": ordinal_soft_attrs,
            "ordinal_soft_sigma": args.ordinal_soft_sigma,
        }
        torch.save(
            {
                "other_state_dict": non_backbone_state_dict(model),
                "config": config,
                "epoch": epoch,
                "metrics": metrics,
            },
            os.path.join(args.lora_save_path, f"{tag}_checkpoint.pt"),
        )
        print(f"Saved {tag} checkpoint to {args.lora_save_path}")

    global_step = 0
    optimizer_step = 0
    best_score = float("-inf")
    optimizer.zero_grad()
    stop = False

    for epoch in range(args.epoch):
        train_loader.dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            outputs = model(**model_inputs(batch), debug_printed=(step + 1 == len(train_loader)))
            loss = outputs["loss"]
            # Normalize the last partial accumulation window by its actual size.
            end_step = min(len(train_loader), args.early_stop_step - epoch * len(train_loader)) if args.early_stop_step > 0 else len(train_loader)
            window_start = (step // args.accumulation_steps) * args.accumulation_steps
            window_size = min(args.accumulation_steps, end_step - window_start)
            (loss / window_size).backward()
            batch_size = batch["input_ids"].size(0)
            running_loss += loss.detach().item() * batch_size
            sample_count += batch_size
            global_step += 1

            should_update = (
                (step + 1) % args.accumulation_steps == 0
                or step + 1 == end_step
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                optimizer_step += 1
                if use_wandb:
                    log = {
                        "train/loss": running_loss / sample_count,
                        "train/optimizer_step": optimizer_step,
                    }
                    for attr in MOVIELENS_ATTRS:
                        log[f"train/loss_{attr}"] = outputs["loss_dict"][f"loss_{attr}"].item()
                    if "w_sigma" in outputs["loss_dict"]:
                        for index, attr in enumerate(MOVIELENS_ATTRS):
                            log[f"train/sigma_{attr}"] = outputs["loss_dict"]["w_sigma"][index].item()
                    for key, value in outputs["loss_dict"].items():
                        if key.startswith("metric_"):
                            log[f"train/{key}"] = value.item()
                    wandb.log(log, step=optimizer_step)

            progress.set_postfix(loss=f"{running_loss / sample_count:.4f}")
            if args.early_stop_step > 0 and global_step >= args.early_stop_step:
                stop = True
                print(f"Early stop at global step {global_step}")
                break

        train_loss = running_loss / max(sample_count, 1)
        val_metrics = evaluate(model, val_loader, device, f"Val {epoch + 1}")
        test_metrics = evaluate(model, test_loader, device, f"Test {epoch + 1}")
        print(f"Epoch {epoch + 1} | train_loss={train_loss:.4f}")
        print("  val :", {key: round(value, 4) for key, value in val_metrics.items()})
        print("  test:", {key: round(value, 4) for key, value in test_metrics.items()})
        if use_wandb:
            wandb.log(
                {**{f"val/{k}": v for k, v in val_metrics.items()},
                 **{f"test/{k}": v for k, v in test_metrics.items()}},
                step=optimizer_step,
            )

        score = val_metrics[args.selection_metric]
        if score > best_score:
            best_score = score
            save_checkpoint(
                "best",
                epoch=epoch,
                metrics={"val": val_metrics, "test": test_metrics},
            )
        if stop:
            break

    save_checkpoint("final", epoch=epoch, metrics=None)
    best = torch.load(os.path.join(args.lora_save_path, "best_checkpoint.pt"), map_location="cpu", weights_only=False)
    from peft import load_peft_weights, set_peft_model_state_dict
    set_peft_model_state_dict(model.backbone, load_peft_weights(
        os.path.join(args.lora_save_path, "best_lora_adapter"), device=str(device)))
    model.load_state_dict(best["other_state_dict"], strict=False)
    print("Best checkpoint test:", evaluate(model, test_loader, device, "Best test",
        os.path.join(args.lora_save_path, "test_distribution.pkl"), save_jsonl=True))
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
