# UGLR

Amazon and MovieLens-1M profile prediction with sqrt-Gini final-query gradient gating, margin-controlled evidence retrieval, and last-token semantic codebook initialization (global centering, norm matching, and a similarity penalty).

`model.py` contains both dataset models. `semantic_init.py` implements shared initialization. Training entry points are `train_amazon.py` and `train_ml1m.py`; `infer_amazon.py` provides standalone Amazon inference.

## Environment

Reference environment: Python 3.11, PyTorch 2.6.0, CUDA 12.4, Transformers 4.51.1, PEFT 0.15.1, and FlashAttention 2.7.4.post1.

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

FlashAttention requires a build environment or wheel compatible with your CUDA and PyTorch versions. Training requires a GPU and access to Llama 3.1 8B Instruct weights.

## Data

See [data_proceessing/README.md](data_proceessing/README.md) for input fields, label mappings, and split reproduction. Run from the repository root:

```bash
# Amazon: reuse existing user IDs and experiment splits.
python data_proceessing/prepare_amazon.py \
  --source-dir /path/to/amazon --reference-dir /path/to/amazon \
  --output-dir data/amazon

# MovieLens-1M: process the three raw .dat files.
python data_proceessing/prepare_ml1m.py \
  --source-dir /path/to/ml-1m --output-dir data/ml1m
```

Both datasets use `dict[user_id, list[str]]` histories and `profiles[user_id]['label_id']` labels. Data files are excluded from Git.

## Training and evaluation

```bash
RPG_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct RPG_GPU_ID=0 bash run_amazon.sh
RPG_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct RPG_GPU_ID=0 bash run_ml1m.sh
```

Optional environment variables: `RPG_PYTHON`, `RPG_SEED` (default: 42), and `RPG_USE_WANDB` (default: 0). Override the Amazon data directory with `RPG_MARGIN_EVIDENCE_DATA_DIR`, or the MovieLens directory with `RPG_DATA_DIR`. Relative paths are resolved from the repository root.

| Setting | Amazon | ML1M |
| --- | --- | --- |
| Attributes/classes | gender 3, education 3, age 5, income 6 | gender 2, occupation 21, age 7 |
| History / token limit | Latest 100 / 8192 | Latest 100 / 8192 |
| Epochs / gradient accumulation | 5 / 8 | 5 / 8 |
| LoRA / head / head-no-WD / codebook LR | 5e-5 / 1e-5 / 1e-4 / 1e-5 | Same |
| Evidence LR | 1e-4 | 1e-4 |
| Loss | Uncertainty weighting; hard labels | Same |
| Best checkpoint | Validation avg_acc (independent decoding by default) | Validation avg_acc |

Amazon runs standalone inference after training. The MovieLens trainer restores the best checkpoint and evaluates it before exiting. Outputs under `lora/` include `best_checkpoint.pt`, `best_lora_adapter/`, `test_distribution.pkl`, and JSONL predictions.

Amazon filters users with `Prefer not to say` labels only from the training set. Validation and test retain these users. Do not remove them during preprocessing.
