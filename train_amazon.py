"""Train the final Amazon UGLR model."""
import math
from functools import partial
from model import AmazonModel
from semantic_init import initialize_codebooks, validate_options
from rpg_history_encoding import HISTORY_PREFIX
import os
import pickle
import argparse
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from rpg_joint_decoder import JointAttributeDecoder, PairwiseAttributePrior
from rpg_metrics import macro_f1_score
from rpg_history_encoding import encode_recent_history


BATCH_SIZE = 1
WEIGHT_DECAY = 1e-2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_VARIANT_DIRS = {
    "unfiltered": os.path.join(
        SCRIPT_DIR, "data", "amazon",
    ),
    "filtered": os.path.join(
        SCRIPT_DIR, "data", "item3core_user_split_his200",
    ),
}


UNKNOWN_LABEL_MAP = {
    "gender": 3,
    "education": 3,
    "income": 6,

}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_train_args(parser):
    'Register training arguments.'
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization, dropout and data shuffling (0..2**32-1).")
    parser.add_argument("--model_path", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--data_variant",
        choices=sorted(DATA_VARIANT_DIRS),
        default="unfiltered",
        help=(
            'Data variant: unfiltered=no item-frequency filtering; '
            'filtered=items with at least three interactions across the full dataset'
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help='Data directory with standard pickle filenames; overrides --data_variant',
    )
    parser.add_argument("--train_file", type=str,
                        default=None,
                        help='Training file; overrides --data_dir/--data_variant')
    parser.add_argument("--valid_file", type=str,
                        default=None,
                        help='Validation file; overrides --data_dir/--data_variant')
    parser.add_argument("--test_file", type=str,
                        default=None,
                        help='Test file; overrides --data_dir/--data_variant')
    parser.add_argument("--label_file", type=str,
                        default=None,
                        help='Label file; overrides --data_dir/--data_variant')
    parser.add_argument("--lora_save_path", type=str,
                        default="./lora/rpg_codebook_direct_query/")

    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5, help='LoRA learning rate')
    parser.add_argument("--head_lr", type=float, default=1e-5,
                        help='Retrieval query-head learning rate')
    parser.add_argument("--head_no_wd_lr", type=float, default=1e-4,
                        help='Learning rate for temperature/log_var scalars without weight decay')
    parser.add_argument("--codebook_lr", type=float, default=5e-4, help='Codebook embedding learning rate')
    parser.add_argument("--aux_lr", type=float, default=1e-4,
                        help='Learning rate for auxiliary_ modules')
    parser.add_argument("--evidence_lr", type=float, default=1e-4,
                        help='Learning rate for evidence_ modules')
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--early_stop_step", type=int, default=50000)
    parser.add_argument(
        "--recent_n", "--max_his_len", dest="max_his_len",
        type=positive_int, default=100,
        help='Keep the most recent N records (default: 100); --max_his_len is an alias',
    )
    parser.add_argument(
        "--max_tokens", "--max_len", dest="max_len",
        type=positive_int, default=8192,
        help='Maximum input tokens including instructions and special tokens (default: 8192); --max_len is an alias',
    )
    parser.add_argument(
        "--query_head_type",
        choices=["direct"],
        default="direct",
        help='The final model uses direct-query heads',
    )


    parser.add_argument("--use_cosine", type=int, default=1,
                        help='1=cosine similarity with CLIP-style logit_scale; 0=dot product/tau')
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learnable_temperature", type=int, default=1)
    parser.add_argument("--loss_mode", type=str, default="uncertainty",
                        choices=["uncertainty", "fixed"],
                        help='uncertainty=Kendall2018 task weighting; fixed=constant weights')
    parser.add_argument("--fixed_weights", type=str, default="1,1,1,1",
                        help='Four fixed weights in gender,education,age,income order')
    parser.add_argument("--ordinal_soft_attrs", type=str, default="age",
                        help='Comma-separated ordinal attributes using soft labels: age/education/income; '
                             "leave empty to disable, e.g. 'age,education,income'")
    parser.add_argument("--ordinal_soft_sigma", type=float, default=1.0,
                        help='Gaussian bandwidth for ordinal soft labels')


    parser.add_argument("--joint_decode", type=int, choices=[0, 1], default=0,
                        help='1=enumerate attribute tuples with a training-set pairwise prior during evaluation')
    parser.add_argument("--joint_prior_mode", choices=["pmi", "joint"], default="pmi",
                        help='pmi=co-occurrence beyond marginal frequencies; joint=log joint probability')
    parser.add_argument("--joint_prior_alpha", type=float, default=1.0,
                        help='Laplace smoothing for pairwise counts')
    parser.add_argument("--joint_prior_clip", type=float, default=3.0,
                        help='Absolute PMI compatibility bound; unused in joint mode')
    parser.add_argument("--joint_decode_beta", type=float, default=0.1,
                        help='Beta in S(u)=sum_j log p_j + beta*C(u)')
    parser.add_argument(
        "--selection_metric", choices=["auto", "avg_acc", "joint_avg_acc"],
        default="auto",
        help='Best-checkpoint metric; auto uses joint_avg_acc when joint decoding is enabled',
    )

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="amazon_profile_rpg_codebook")
    parser.add_argument("--wandb_run_name", type=str,
                        default="llama3_rpg_codebook_direct_query")
    parser.add_argument(
        "--optimizer_diagnostics_interval", type=int, default=1,
        help=(
            'Optimizer-step interval for full parameter-update diagnostics; '
            '0=only the last step of each epoch'
        ),
    )
    return parser


def resolve_data_paths(args):
    'Resolve data paths while allowing explicit per-file overrides.'
    variant = getattr(args, "data_variant", "unfiltered")
    data_dir = getattr(args, "data_dir", None) or DATA_VARIANT_DIRS[variant]
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    default_filenames = {
        "train_file": "train_user_history_his200.pkl",
        "valid_file": "val_user_history_his200.pkl",
        "test_file": "test_user_history_his200.pkl",
        "label_file": "true_user_profile.pkl",
    }

    for argument, filename in default_filenames.items():
        path = getattr(args, argument, None) or os.path.join(data_dir, filename)
        setattr(args, argument, os.path.abspath(os.path.expanduser(path)))

    missing = [
        f"{argument}={getattr(args, argument)}"
        for argument in default_filenames
        if not os.path.isfile(getattr(args, argument))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing data file(s): " + ", ".join(missing)
        )

    args.data_dir = data_dir
    return args


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_num_classes(label_data, key):
    return max(x[key] for x in label_data) + 1


def move_batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class CustomDataset(Dataset):
    'A multi-task sample contains age, gender, education and income labels. History records are joined into one prompt.'

    def __init__(self, history_items, labels, sample_keys, tokenizer,
                 max_his_len=100, max_len=8192, pad_to_max_length=False):
        self.history_items = history_items
        self.labels = labels
        self.sample_keys = sample_keys
        self.tokenizer = tokenizer
        self.max_his_len = max_his_len
        self.max_len = max_len
        self.pad_to_max_length = pad_to_max_length

    def __len__(self):
        return len(self.history_items)

    def __getitem__(self, idx):
        _, prompt, encoding = encode_recent_history(
            self.tokenizer, self.history_items[idx], self.max_his_len, self.max_len,
            pad_to_max_length=self.pad_to_max_length,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = self.labels[idx]
        return {
            "sample_key": self.sample_keys[idx],
            "prompt": prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "age": torch.tensor(label["age"], dtype=torch.long),
            "gender": torch.tensor(label["gender"], dtype=torch.long),
            "education": torch.tensor(label["education"], dtype=torch.long),
            "income": torch.tensor(label["income"], dtype=torch.long),
        }


def has_unknown_label(label_ids):
    'Check whether a user has an unknown label outside the model output classes.'
    return any(label_ids[attr] == unknown_id
               for attr, unknown_id in UNKNOWN_LABEL_MAP.items())


def build_multitask_data(input_file_path, label_file, keys, filter_unknown=True):
    'Build multi-task data, filtering unknown training labels while retaining evaluation users.'
    input_file = load_pickle(input_file_path)
    inputs, labels, sample_keys = [], [], []
    for k in keys:
        label_ids = label_file[k]["label_id"]
        if filter_unknown and has_unknown_label(label_ids):
            continue
        inputs.append(input_file[k])
        labels.append(label_ids)
        sample_keys.append(k)
    return inputs, labels, sample_keys


@torch.no_grad()
def evaluate(model, loader, device, desc="Evaluating", joint_decoder=None):
    model.eval()
    total = 0
    correct = {"age": 0, "gender": 0, "education": 0, "income": 0}
    all_targets = {attr: [] for attr in correct}
    all_predictions = {attr: [] for attr in correct}
    joint_correct = {attr: 0 for attr in correct} if joint_decoder else None
    joint_predictions = {attr: [] for attr in correct} if joint_decoder else None
    tuple_correct = 0
    joint_tuple_correct = 0
    joint_changed_users = 0
    joint_improved_attributes = 0
    joint_harmed_attributes = 0
    total_loss = 0.0
    loss_count = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch_to_device(batch, device)


        batch_has_unknown = any(
            (batch[attr] == unknown_id).any().item()
            for attr, unknown_id in UNKNOWN_LABEL_MAP.items()
        )
        model_kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        for key in ("item_end_positions", "item_mask"):
            if key in batch:
                model_kwargs[key] = batch[key]
        if not batch_has_unknown:
            model_kwargs.update({
                "age": batch["age"],
                "gender": batch["gender"],
                "education": batch["education"],
                "income": batch["income"],
            })
        outputs = model(**model_kwargs)
        bs = batch["input_ids"].size(0)
        total += bs
        logits_map = {
            attr: outputs[f"{attr}_logits"].float()
            for attr in correct
        }
        independent_preds = {
            attr: logits_map[attr].argmax(dim=-1)
            for attr in correct
        }
        for attr in correct:
            pred = independent_preds[attr]
            correct[attr] += (pred == batch[attr]).sum().item()
            all_predictions[attr].extend(pred.detach().cpu().tolist())
            all_targets[attr].extend(batch[attr].detach().cpu().tolist())
        independent_matrix = torch.stack(
            [independent_preds[attr] for attr in model.attr_order], dim=1
        )
        target_matrix = torch.stack(
            [batch[attr] for attr in model.attr_order], dim=1
        )
        tuple_correct += int(independent_matrix.eq(target_matrix).all(dim=1).sum().item())

        if joint_decoder is not None:
            decoded = joint_decoder.decode(logits_map)
            for attr in correct:
                joint_correct[attr] += int((decoded[attr] == batch[attr]).sum().item())
                joint_predictions[attr].extend(decoded[attr].detach().cpu().tolist())
            joint_matrix = torch.stack(
                [decoded[attr] for attr in model.attr_order], dim=1
            )
            independent_is_correct = independent_matrix.eq(target_matrix)
            joint_is_correct = joint_matrix.eq(target_matrix)
            joint_tuple_correct += int(joint_is_correct.all(dim=1).sum().item())
            joint_changed_users += int(
                joint_matrix.ne(independent_matrix).any(dim=1).sum().item()
            )
            joint_improved_attributes += int(
                (joint_is_correct & ~independent_is_correct).sum().item()
            )
            joint_harmed_attributes += int(
                (~joint_is_correct & independent_is_correct).sum().item()
            )
        if outputs["loss"] is not None:
            total_loss += outputs["loss"].item() * bs
            loss_count += bs

    metrics = {f"{attr}_acc": correct[attr] / total for attr in correct}
    metrics["avg_acc"] = sum(correct.values()) / (4 * total)
    for attr in correct:
        metrics[f"{attr}_macro_f1"] = macro_f1_score(
            all_targets[attr], all_predictions[attr]
        )
    metrics["avg_macro_f1"] = sum(
        metrics[f"{attr}_macro_f1"] for attr in correct
    ) / len(correct)
    metrics["tuple_acc"] = tuple_correct / total
    if joint_decoder is not None:
        for attr in correct:
            metrics[f"joint_{attr}_acc"] = joint_correct[attr] / total
            metrics[f"joint_{attr}_macro_f1"] = macro_f1_score(
                all_targets[attr], joint_predictions[attr]
            )
        metrics["joint_avg_acc"] = sum(joint_correct.values()) / (4 * total)
        metrics["joint_avg_macro_f1"] = sum(
            metrics[f"joint_{attr}_macro_f1"] for attr in correct
        ) / len(correct)
        metrics["joint_tuple_acc"] = joint_tuple_correct / total
        metrics["joint_changed_users"] = joint_changed_users
        metrics["joint_changed_rate"] = joint_changed_users / total
        metrics["joint_improved_attributes"] = joint_improved_attributes
        metrics["joint_harmed_attributes"] = joint_harmed_attributes

    metrics["loss"] = total_loss / loss_count if loss_count > 0 else 0.0
    metrics["loss_count"] = loss_count
    return metrics


def get_non_backbone_state_dict(model, backbone_prefix="backbone."):
    return {k: v.cpu() for k, v in model.state_dict().items()
            if not k.startswith(backbone_prefix)}


@torch.no_grad()
def get_codebook_diagnostics(model, initial_codebooks):
    """Return direction drift and inter-value cosine similarity per codebook.

    ``direction_change`` is mean(1 - cosine(current_value, initial_value)).
    ``value_cosine_mean`` is the mean cosine over distinct value pairs in the
    current codebook (the diagonal/self-similarity is excluded).
    """
    diagnostics = {}
    for attr in model.attr_order:
        current = getattr(model, f"{attr}_head").codebook.weight.detach().float()
        initial = initial_codebooks[attr]
        current_normalized = torch.nn.functional.normalize(current, dim=-1)
        initial_normalized = torch.nn.functional.normalize(initial, dim=-1)

        diagnostics[attr] = {
            "direction_change": (
                1.0 - (current_normalized * initial_normalized).sum(dim=-1)
            ).mean().item(),
        }

        num_values = current_normalized.size(0)
        if num_values > 1:
            cosine_matrix = current_normalized @ current_normalized.t()
            off_diagonal_mask = ~torch.eye(
                num_values, dtype=torch.bool, device=cosine_matrix.device
            )
            value_cosine_mean = cosine_matrix[off_diagonal_mask].mean().item()
        else:
            value_cosine_mean = 0.0
        diagnostics[attr]["value_cosine_mean"] = value_cosine_mean

    return diagnostics


@torch.no_grad()
def get_optimizer_group_grad_norms(optimizer):
    """Compute the L2 norm of gradients in each named optimizer group."""
    norms = {}
    for group in optimizer.param_groups:
        squared_norm = 0.0
        for param in group["params"]:
            if param.grad is None:
                continue
            param_grad_norm = param.grad.detach().float().norm().item()
            squared_norm += param_grad_norm * param_grad_norm
        norms[group["group_name"]] = squared_norm ** 0.5
    return norms


@torch.no_grad()
def capture_optimizer_group_parameters(optimizer):
    """Snapshot parameters that will be updated and their pre-step group norms."""
    snapshots = {}
    parameter_norms = {}
    for group in optimizer.param_groups:
        group_name = group["group_name"]
        group_snapshots = []
        squared_norm = 0.0
        for param in group["params"]:
            # AdamW skips parameters whose gradients are None, including weight decay.
            if param.grad is None:
                continue
            before = param.detach().clone()
            group_snapshots.append((param, before))
            param_norm = before.float().norm().item()
            squared_norm += param_norm * param_norm
        snapshots[group_name] = group_snapshots
        parameter_norms[group_name] = squared_norm ** 0.5
    return snapshots, parameter_norms


@torch.no_grad()
def get_optimizer_group_update_metrics(snapshots, parameter_norms):
    """Measure exact optimizer-step deltas per group from parameter snapshots."""
    metrics = {}
    for group_name, group_snapshots in snapshots.items():
        squared_update_norm = 0.0
        for param, before in group_snapshots:
            update_norm = (param.detach().float() - before.float()).norm().item()
            squared_update_norm += update_norm * update_norm
        group_update_norm = squared_update_norm ** 0.5
        group_parameter_norm = parameter_norms[group_name]
        metrics[group_name] = {
            "parameter_norm": group_parameter_norm,
            "update_norm": group_update_norm,
            "update_to_parameter_ratio": (
                group_update_norm / group_parameter_norm
                if group_parameter_norm > 0.0 else 0.0
            ),
        }
    return metrics


def _train(args):
    """Train the final Amazon model and preserve the original optimizer/checkpoint policy."""
    model_class = partial(AmazonModel, **{key: getattr(args, key) for key in EVIDENCE_KEYS})
    model_setup_fn = describe_variant
    variant_name = VARIANT  # Retained for checkpoint compatibility.
    train_model_input_keys = ('item_end_positions', 'item_mask', 'gate_epoch')
    dataset_factory_fn = dataset_factory
    args.seed = getattr(args, "seed", 42)
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32-1]")
    # Seed before constructing datasets, LoRA adapters or classification heads.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed: {args.seed}")
    args = resolve_data_paths(args)
    print(
        f"Data variant: {getattr(args, 'data_variant', 'unfiltered')}\n"
        f"  train: {args.train_file}\n"
        f"  valid: {args.valid_file}\n"
        f"  test:  {args.test_file}\n"
        f"  label: {args.label_file}"
    )
    use_wandb = bool(args.use_wandb)
    if use_wandb:
        import wandb

    os.makedirs(args.lora_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Llama 3 has no pad token and falls back to EOS.  Qwen3.5 ships a real
    # <|endoftext|> pad token, which must not be replaced by its <|im_end|> EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    label_file = load_pickle(args.label_file)
    train_keys = list(load_pickle(args.train_file).keys())
    val_keys = list(load_pickle(args.valid_file).keys())
    test_keys = list(load_pickle(args.test_file).keys())


    train_x, train_y, train_k = build_multitask_data(
        args.train_file, label_file, train_keys, filter_unknown=True,
    )
    val_x, val_y, val_k = build_multitask_data(
        args.valid_file, label_file, val_keys, filter_unknown=False,
    )
    test_x, test_y, test_k = build_multitask_data(
        args.test_file, label_file, test_keys, filter_unknown=False,
    )
    print(f"train={len(train_x)} (unknown filtered), "
          f"val={len(val_x)}, test={len(test_x)} (full evaluation sets)")

    def make_loader(x, y, k, shuffle, split):
        ds = dataset_factory_fn(
            split=split, history_items=x, labels=y, sample_keys=k,
            tokenizer=tokenizer, args=args,
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_x, train_y, train_k, True, "train")
    val_loader = make_loader(val_x, val_y, val_k, False, "val")
    test_loader = make_loader(test_x, test_y, test_k, False, "test")

    num_gender = get_num_classes(train_y, "gender")
    num_education = get_num_classes(train_y, "education")
    num_age = get_num_classes(train_y, "age")
    num_income = get_num_classes(train_y, "income")
    print(f"num_classes: gender={num_gender}, education={num_education}, "
          f"age={num_age}, income={num_income}")

    num_classes_map = {
        "gender": num_gender,
        "education": num_education,
        "age": num_age,
        "income": num_income,
    }
    joint_decoder = None
    joint_decoder_config = None
    if bool(args.joint_decode):
        pairwise_prior = PairwiseAttributePrior.fit(
            labels=train_y,
            num_classes=num_classes_map,
            alpha=args.joint_prior_alpha,
            mode=args.joint_prior_mode,
            clip=args.joint_prior_clip,
        )
        joint_decoder = JointAttributeDecoder(
            prior=pairwise_prior,
            beta=args.joint_decode_beta,
        )
        joint_decoder_config = joint_decoder.to_dict(include_tables=True)
        print(
            "Joint decoder enabled:",
            {**joint_decoder.to_dict(include_tables=False),
             "prior": pairwise_prior.to_dict(include_tables=False)},
        )

    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = "joint_avg_acc" if joint_decoder is not None else "avg_acc"
    if selection_metric == "joint_avg_acc" and joint_decoder is None:
        raise ValueError("selection_metric=joint_avg_acc requires --joint_decode 1")
    print("Best-checkpoint selection metric:", selection_metric)

    # ===== backbone + LoRA =====
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=None,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )

    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=lora_target_modules,
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device=device, dtype=torch.bfloat16)

    hidden_size = base_model.config.hidden_size
    fixed_weights = [float(w) for w in args.fixed_weights.split(",")]
    ordinal_soft_attrs = [a.strip() for a in args.ordinal_soft_attrs.split(",") if a.strip()]
    print(f"ordinal_soft_attrs={ordinal_soft_attrs}, sigma={args.ordinal_soft_sigma}")

    model = model_class(
        peft_causal_lm=base_model,
        hidden_size=hidden_size,
        num_gender_classes=num_gender,
        num_education_classes=num_education,
        num_age_classes=num_age,
        num_income_classes=num_income,
        use_cosine=bool(args.use_cosine),
        temperature=args.temperature,
        learnable_temperature=bool(args.learnable_temperature),
        loss_mode=args.loss_mode,
        fixed_weights=fixed_weights,
        ordinal_soft_attrs=ordinal_soft_attrs,
        ordinal_soft_sigma=args.ordinal_soft_sigma,
    )
    model = model.set_mixed_precision(device=device, backbone_dtype=torch.bfloat16)
    dtype_info = model.dtype_info()
    print("Mixed-precision dtypes:", dtype_info)

    model_setup = {}
    if model_setup_fn is not None:
        model_setup = model_setup_fn(model, tokenizer, device, args) or {}
        print(f"Model variant setup ({variant_name}):", model_setup)

    # Capture the actual training starting point after variant-specific setup.
    # Keeping these tiny tensors on-device avoids a CPU->GPU copy at every log step.
    initial_codebooks = {
        attr: getattr(model, f"{attr}_head").codebook.weight.detach().float().clone()
        for attr in model.attr_order
    }


    lora_params, head_params_wd, head_params_no_wd = [], [], []
    codebook_params, auxiliary_params, evidence_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            lora_params.append(param)
        elif "codebook" in name:
            codebook_params.append(param)
        elif "auxiliary_" in name:
            auxiliary_params.append(param)
        elif "evidence_" in name:
            evidence_params.append(param)
        elif ("log_logit_scale" in name) or ("log_tau" in name) or ("log_var" in name):

            head_params_no_wd.append(param)
        else:
            head_params_wd.append(param)
    print(f"LoRA params: {len(lora_params)}, "
          f"Head(wd) params: {len(head_params_wd)}, "
          f"Head(no_wd) params: {len(head_params_no_wd)}, "
          f"Codebook params: {len(codebook_params)}, "
          f"Auxiliary params: {len(auxiliary_params)}, "
          f"Evidence params: {len(evidence_params)}")

    optimizer_group_specs = [
        {
            "group_name": "lora",
            "params": lora_params,
            "lr": args.lr,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "group_name": "head_wd",
            "params": head_params_wd,
            "lr": args.head_lr,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "group_name": "head_no_wd",
            "params": head_params_no_wd,
            "lr": args.head_no_wd_lr,
            "weight_decay": 0.0,
        },
        {
            "group_name": "codebook",
            "params": codebook_params,
            "lr": args.codebook_lr,
            "weight_decay": 0.0,
        },
    ]
    if auxiliary_params:
        optimizer_group_specs.append({
            "group_name": "auxiliary",
            "params": auxiliary_params,
            "lr": args.aux_lr,
            "weight_decay": WEIGHT_DECAY,
        })
    if evidence_params:
        optimizer_group_specs.append({
            "group_name": "evidence",
            "params": evidence_params,
            "lr": args.evidence_lr,
            "weight_decay": WEIGHT_DECAY,
        })
    optimizer = torch.optim.AdamW(optimizer_group_specs)

    optimizer_group_config = []
    for group in optimizer.param_groups:
        group_info = {
            "group_name": group["group_name"],
            "initial_lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "num_parameter_tensors": len(group["params"]),
            "num_parameters": sum(p.numel() for p in group["params"]),
        }
        optimizer_group_config.append(group_info)
        print("Optimizer group:", group_info)

    accumulation_steps = args.accumulation_steps

    total_steps = max((len(train_loader) // accumulation_steps) * args.epoch, 1)
    warmup_steps = max(total_steps // 10, 1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if use_wandb:
        wandb_config = vars(args).copy()
        wandb_config["optimizer_param_groups"] = optimizer_group_config
        wandb_config["mixed_precision_dtypes"] = dtype_info
        wandb_config["model_variant"] = variant_name
        wandb_config["model_setup"] = model_setup
        wandb_config["selection_metric_resolved"] = selection_metric
        wandb_config["joint_decoder"] = joint_decoder_config
        wandb_config["training_stages"] = None
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   config=wandb_config)

    def save_ckpt(save_dir, tag, extra=None):
        os.makedirs(save_dir, exist_ok=True)
        lora_dir = os.path.join(save_dir, f"{tag}_lora_adapter")
        model.backbone.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)
        ckpt = {
            "other_state_dict": get_non_backbone_state_dict(model),
            "config": {
                "seed": args.seed,
                "hidden_size": hidden_size,
                "num_gender_classes": num_gender,
                "num_education_classes": num_education,
                "num_age_classes": num_age,
                "num_income_classes": num_income,
                "use_cosine": bool(args.use_cosine),
                "temperature": args.temperature,
                "learnable_temperature": bool(args.learnable_temperature),
                "loss_mode": args.loss_mode,
                "fixed_weights": fixed_weights,
                "ordinal_soft_attrs": ordinal_soft_attrs,
                "ordinal_soft_sigma": args.ordinal_soft_sigma,
                "mixed_precision_dtypes": dtype_info,
                "model_variant": variant_name,
                "model_setup": model_setup,
                "selection_metric": selection_metric,
                "joint_decoder": joint_decoder_config,
            },
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, os.path.join(save_dir, f"{tag}_checkpoint.pt"))
        print(f"Saved {tag} checkpoint to {save_dir}")

    global_step, optimizer_step, best_val_score = 0, 0, -1.0

    for epoch in range(args.epoch):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        running_loss, count = 0.0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")

        for step, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, device)

            debug_flag = (step + 1) == len(train_loader)
            extra_model_kwargs = {
                key: batch[key]
                for key in train_model_input_keys
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                age=batch["age"], gender=batch["gender"],
                education=batch["education"], income=batch["income"],
                debug_printed=debug_flag,
                **extra_model_kwargs,
            )
            loss = outputs["loss"]
            raw_loss = loss.detach().item()
            (loss / accumulation_steps).backward()

            bs = batch["input_ids"].size(0)
            running_loss += raw_loss * bs
            count += bs
            global_step += 1

            do_update = ((step + 1) % accumulation_steps == 0) or ((step + 1) == len(train_loader))
            if do_update:
                diagnostics_interval = args.optimizer_diagnostics_interval
                monitor_optimizer_groups = debug_flag or (
                    use_wandb
                    and diagnostics_interval > 0
                    and (optimizer_step + 1) % diagnostics_interval == 0
                )
                group_grad_norms_raw = None
                group_grad_norms_clipped = None
                group_parameter_snapshots = None
                group_parameter_norms = None
                if monitor_optimizer_groups:
                    group_grad_norms_raw = get_optimizer_group_grad_norms(optimizer)
                    group_parameter_snapshots, group_parameter_norms = (
                        capture_optimizer_group_parameters(optimizer)
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if monitor_optimizer_groups:
                    group_grad_norms_clipped = get_optimizer_group_grad_norms(optimizer)

                optimizer.step()
                group_update_metrics = None
                if monitor_optimizer_groups:
                    group_update_metrics = get_optimizer_group_update_metrics(
                        group_parameter_snapshots, group_parameter_norms
                    )
                    del group_parameter_snapshots
                optimizer.zero_grad()
                scheduler.step()
                optimizer_step += 1

                codebook_diagnostics = None
                if use_wandb or debug_flag:
                    codebook_diagnostics = get_codebook_diagnostics(
                        model, initial_codebooks
                    )

                if debug_flag and codebook_diagnostics is not None:
                    print("Codebook direction diagnostics (after optimizer update):")
                    for attr in model.attr_order:
                        diag = codebook_diagnostics[attr]
                        print(
                            f"  [{attr}] direction_change="
                            f"{diag['direction_change']:.6f}, "
                            f"value_cosine_mean={diag['value_cosine_mean']:.6f}"
                        )

                if debug_flag and group_update_metrics is not None:
                    print("Optimizer-group diagnostics:")
                    for group_name in group_update_metrics:
                        update_metrics = group_update_metrics[group_name]
                        print(
                            f"  [{group_name}] grad_norm_raw="
                            f"{group_grad_norms_raw[group_name]:.6e}, "
                            f"grad_norm_clipped="
                            f"{group_grad_norms_clipped[group_name]:.6e}, "
                            f"update_norm={update_metrics['update_norm']:.6e}, "
                            f"update/parameter="
                            f"{update_metrics['update_to_parameter_ratio']:.6e}"
                        )

                if use_wandb:
                    ld = outputs["loss_dict"]
                    optimizer_groups = {
                        group["group_name"]: group
                        for group in optimizer.param_groups
                    }
                    log = {
                        "train/loss": running_loss / count,
                        "train/lr_lora": optimizer_groups["lora"]["lr"],

                        "train/lr_head": optimizer_groups["head_wd"]["lr"],
                        "train/lr_head_no_wd": optimizer_groups["head_no_wd"]["lr"],
                        "train/lr_codebook": optimizer_groups["codebook"]["lr"],
                        "train/weight_decay_lora": optimizer_groups["lora"]["weight_decay"],
                        "train/weight_decay_head_wd": optimizer_groups["head_wd"]["weight_decay"],
                        "train/weight_decay_head_no_wd": optimizer_groups["head_no_wd"]["weight_decay"],
                        "train/weight_decay_codebook": optimizer_groups["codebook"]["weight_decay"],
                    }
                    if "auxiliary" in optimizer_groups:
                        log["train/lr_auxiliary"] = optimizer_groups["auxiliary"]["lr"]
                        log["train/weight_decay_auxiliary"] = (
                            optimizer_groups["auxiliary"]["weight_decay"]
                        )
                    if "evidence" in optimizer_groups:
                        log["train/lr_evidence"] = optimizer_groups["evidence"]["lr"]
                        log["train/weight_decay_evidence"] = (
                            optimizer_groups["evidence"]["weight_decay"]
                        )
                    for attr in ["gender", "education", "age", "income"]:
                        if f"loss_{attr}" in ld:
                            log[f"train/loss_{attr}"] = ld[f"loss_{attr}"].item()
                    for key, value in ld.items():
                        metric_name = f"train/{key}"
                        if (
                            key.startswith(("loss_", "metric_"))
                            and metric_name not in log
                        ):
                            log[metric_name] = value.item()
                    if "w_sigma" in ld:
                        for i, attr in enumerate(["gender", "education", "age", "income"]):
                            log[f"train/sigma_{attr}"] = ld["w_sigma"][i].item()


                    if bool(args.use_cosine):
                        for attr in ["gender", "education", "age", "income"]:
                            head = getattr(model, f"{attr}_head")
                            ls = head._logit_scale().item()
                            log[f"train/logit_scale_{attr}"] = ls
                            log[f"train/codebook_norm_{attr}"] = (
                                head.codebook.weight.detach().float().norm(dim=-1).mean().item()
                            )
                    for attr in ["gender", "education", "age", "income"]:
                        diag = codebook_diagnostics[attr]
                        log[f"train/codebook_direction_change_{attr}"] = (
                            diag["direction_change"]
                        )
                        log[f"train/codebook_value_cosine_mean_{attr}"] = (
                            diag["value_cosine_mean"]
                        )
                    if group_update_metrics is not None:
                        for group_name, update_metrics in group_update_metrics.items():
                            log[f"train/grad_norm_raw_{group_name}"] = (
                                group_grad_norms_raw[group_name]
                            )
                            log[f"train/grad_norm_clipped_{group_name}"] = (
                                group_grad_norms_clipped[group_name]
                            )
                            log[f"train/parameter_norm_{group_name}"] = (
                                update_metrics["parameter_norm"]
                            )
                            log[f"train/update_norm_{group_name}"] = (
                                update_metrics["update_norm"]
                            )
                            log[f"train/update_to_parameter_ratio_{group_name}"] = (
                                update_metrics["update_to_parameter_ratio"]
                            )
                    wandb.log(log, step=optimizer_step)

            pbar.set_postfix({"loss": f"{running_loss / count:.4f}", "opt_step": optimizer_step})

            if global_step >= args.early_stop_step:
                print(f"Early stop at global step {global_step}")
                break

            del outputs
            if torch.cuda.is_available() and step % 500 == 0:
                torch.cuda.empty_cache()

        epoch_loss = running_loss / max(count, 1)
        val_metrics = evaluate(
            model, val_loader, device, desc=f"Val {epoch + 1}",
            joint_decoder=joint_decoder,
        )
        test_metrics = evaluate(
            model, test_loader, device, desc=f"Test {epoch + 1}",
            joint_decoder=joint_decoder,
        )
        print(f"Epoch {epoch + 1} | train_loss={epoch_loss:.4f}")
        print("  val :", {k: round(v, 4) for k, v in val_metrics.items()})
        print("  test:", {k: round(v, 4) for k, v in test_metrics.items()})

        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=optimizer_step)
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=optimizer_step)

        if val_metrics[selection_metric] > best_val_score:
            best_val_score = val_metrics[selection_metric]
            save_ckpt(args.lora_save_path, "best",
                      extra={"epoch": epoch, "val_metrics": val_metrics,
                             "test_metrics": test_metrics,
                             "selection_metric": selection_metric,
                             "best_selection_score": best_val_score,

                             "best_val_acc": val_metrics["avg_acc"]})

        if global_step >= args.early_stop_step:
            break

    save_ckpt(args.lora_save_path, "final")
    if use_wandb:
        wandb.finish()


EVIDENCE_KEYS = ("gate_min", "gate_loss_scale", "gate_warmup_epochs", "evidence_dim", "rho_min", "rho_max", "initial_loss_beta")
VARIANT = "rpg_codebook_sqrt_gini_margin_evidence_semantic_spread_final_query_gate"

class MarginEvidenceDataset(CustomDataset):
    """Adds fixed-width end-token positions for complete, visible history items."""
    epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, idx):
        items, prompt, encoded = encode_recent_history(
            self.tokenizer, self.history_items[idx], self.max_his_len, self.max_len,
            return_offsets_mapping=True,
        )
        spans, cursor = [], len(HISTORY_PREFIX)
        for item in items:
            spans.append((cursor, cursor + len(item)))
            cursor += len(item) + 1
        input_ids, attention_mask = encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)
        offsets = encoded['offset_mapping'].squeeze(0).tolist()
        ends, visible = [], []
        visible_end = max((end for start, end in offsets if end > start), default=0)
        for start, end in spans:
            token_ids = [i for i, (left, right) in enumerate(offsets)
                         if right > left and left < end and right > start]
            complete = bool(token_ids) and end <= visible_end
            ends.append(token_ids[-1] if complete else 0)
            visible.append(complete)
        # A fixed N keeps the default physical batch=1 contract explicit and also collates safely.
        pad = self.max_his_len - len(ends)
        result = {
            'sample_key': self.sample_keys[idx], 'prompt': prompt,
            'input_ids': input_ids, 'attention_mask': attention_mask,
            'age': torch.tensor(self.labels[idx]['age'], dtype=torch.long),
            'gender': torch.tensor(self.labels[idx]['gender'], dtype=torch.long),
            'education': torch.tensor(self.labels[idx]['education'], dtype=torch.long),
            'income': torch.tensor(self.labels[idx]['income'], dtype=torch.long),
            'item_end_positions': torch.tensor(ends + [0] * pad, dtype=torch.long),
            'item_mask': torch.tensor(visible + [False] * pad, dtype=torch.bool),
            'gate_epoch': torch.tensor(self.epoch, dtype=torch.long),
        }
        return result

def dataset_factory(split, history_items, labels, sample_keys, tokenizer, args):
    return MarginEvidenceDataset(history_items, labels, sample_keys, tokenizer,
                                 max_his_len=args.max_his_len, max_len=args.max_len)

def describe_variant(model, tokenizer, device, args):
    setup = dict(**{key: getattr(args, key) for key in EVIDENCE_KEYS},
                query_head_type='direct_attribute_projection',
                uncertainty_source=f'detached_final_current_{model.gate_uncertainty_metric}_gate_and_initial_top1_top2_margin_rho',
                gate_uncertainty_metric=model.gate_uncertainty_metric, gate_uncertainty_normalized=True,
                gate_uncertainty_formula=model.gate_uncertainty_formula,
                evidence_uncertainty_metric='margin',
                gate_formula='g_min+(1-g_min)*(1-u)/(1+u/gate_loss_scale)',
                rho_formula='rho_min+(rho_max-rho_min)*u',
                initial_probability_source='train_mode_q0_logits',
                gate_probability_source='train_mode_q1_logits',
                initial_final_loss_mix=f'{1-args.initial_loss_beta}:final,{args.initial_loss_beta}:initial',
                item_visibility='complete_untruncated_item_end_tokens_only',
                memory_detach='before_FP32_memory_norm_and_projection',
                attention='shared_single_head_attribute_query',
                output_projection='per_attribute_zero_initialized',
                task_loss_weighting=args.loss_mode, gate_detached=True, rho_detached=True,
                gate_task_order=model.attr_order, backbone_forwards_per_microbatch=1,
                inference_prediction='final_logits')
    setup.update(initialize_codebooks(model, tokenizer, device, args, spread=True))
    return setup

def build_parser():
    parser = add_train_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(ordinal_soft_attrs='', ordinal_soft_sigma=.5,
                        lora_save_path='./lora/rpg_codebook_sqrt_gini_margin_evidence_semantic_spread_final_query_gate/',
                        wandb_run_name='llama3_rpg_codebook_sqrt_gini_margin_evidence_semantic_spread_final_query_gate')
    parser.add_argument('--gate_min', type=float, default=.1)
    parser.add_argument('--gate_loss_scale', type=float, default=.5)
    parser.add_argument('--gate_warmup_epochs', type=int, default=1)
    parser.add_argument('--evidence_dim', type=int, default=128)
    parser.add_argument('--rho_min', type=float, default=.05)
    parser.add_argument('--rho_max', type=float, default=.5)
    parser.add_argument('--initial_loss_beta', type=float, default=.2)
    parser.add_argument('--semantic_center_strength', type=float, default=1.)
    parser.add_argument('--semantic_similarity_weight', type=float, default=.1)
    parser.add_argument('--semantic_similarity_steps', type=int, default=200)
    parser.add_argument('--semantic_similarity_lr', type=float, default=.05)
    parser.add_argument('--semantic_within_cap', type=float, default=.5)
    parser.add_argument('--semantic_cross_cap', type=float, default=.3)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.query_head_type != 'direct': parser.error('The final model requires --query_head_type direct')
    if not 0 <= args.gate_min <= 1 or not math.isfinite(args.gate_loss_scale) or args.gate_loss_scale <= 0:
        parser.error('invalid gate parameters')
    if args.gate_warmup_epochs < 0 or args.evidence_dim <= 0 or not 0 <= args.rho_min <= args.rho_max:
        parser.error('invalid evidence parameters')
    if not 0 <= args.initial_loss_beta <= 1: parser.error('initial_loss_beta must be in [0,1]')
    _train(args)


if __name__ == "__main__":
    main()
