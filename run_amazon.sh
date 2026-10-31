#!/usr/bin/env bash
# Final-query sqrt-Gini gradient gate; last-token semantic initialization; global centering and norm matching; similarity penalty=True.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${RPG_PYTHON:-python}"
RPG_SEED="${RPG_SEED:-42}"  # Override with RPG_SEED=123
export PYTHONHASHSEED="$RPG_SEED"
MODEL_PATH="${RPG_MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATA_DIR="${RPG_MARGIN_EVIDENCE_DATA_DIR:-./data/amazon}"
LOSS_MODE="${RPG_LOSS_MODE:-uncertainty}"
SAVE_DIR="${RPG_SQRT_GINI_MARGIN_EVIDENCE_SEMANTIC_SPREAD_FINAL_QUERY_GATE_SAVE_DIR:-./lora/rpg_codebook_sqrt_gini_margin_evidence_semantic_spread_final_query_gate_${LOSS_MODE}_hardlabel_edu3_age5_unfiltered_his100_tokens8192}"
RECENT_N=100
MAX_TOKENS=8192
export CUDA_VISIBLE_DEVICES="${RPG_GPU_ID:-${CUDA_VISIBLE_DEVICES:-1}}"

"$PYTHON_BIN" train_amazon.py \
    --seed "$RPG_SEED" \
    --model_path "$MODEL_PATH" --data_variant unfiltered --data_dir "$DATA_DIR" \
    --recent_n "$RECENT_N" --max_tokens "$MAX_TOKENS" \
    --epoch 5 --lr 5e-5 --head_lr 1e-5 --head_no_wd_lr 1e-4 --codebook_lr 1e-5 \
    --evidence_lr "${RPG_EVIDENCE_LR:-1e-4}" --accumulation_steps 8 \
    --use_cosine 1 --temperature .07 --learnable_temperature 1 --query_head_type direct \
    --loss_mode "$LOSS_MODE" --fixed_weights 1,1,1,1 \
    --ordinal_soft_attrs "" \
    --gate_warmup_epochs "${RPG_GATE_WARMUP_EPOCHS:-1}" --gate_min "${RPG_GATE_MIN:-0.1}" \
    --gate_loss_scale "${RPG_GATE_LOSS_SCALE:-0.5}" --evidence_dim "${RPG_EVIDENCE_DIM:-128}" \
    --rho_min "${RPG_RHO_MIN:-0.05}" --rho_max "${RPG_RHO_MAX:-0.5}" \
    --initial_loss_beta "${RPG_INITIAL_LOSS_BETA:-0.2}" \
    --semantic_center_strength "${RPG_SEMANTIC_CENTER_STRENGTH:-1.0}" \
    --semantic_similarity_weight "${RPG_SEMANTIC_SIMILARITY_WEIGHT:-0.1}" \
    --semantic_similarity_steps "${RPG_SEMANTIC_SIMILARITY_STEPS:-200}" \
    --semantic_similarity_lr "${RPG_SEMANTIC_SIMILARITY_LR:-0.05}" \
    --semantic_within_cap "${RPG_SEMANTIC_WITHIN_CAP:-0.5}" \
    --semantic_cross_cap "${RPG_SEMANTIC_CROSS_CAP:-0.3}" \
    --lora_save_path "$SAVE_DIR" --use_wandb "${RPG_USE_WANDB:-0}" \
    --wandb_project amazon_profile_edu3_age5 \
    --wandb_run_name "llama3_rpg_codebook_sqrt_gini_margin_evidence_semantic_spread_final_query_gate_${LOSS_MODE}_hardlabel_unfiltered_his100_tokens8192"

"$PYTHON_BIN" infer_amazon.py \
    --model_path "$MODEL_PATH" --test_file "$DATA_DIR/test_user_history_his200.pkl" \
    --label_file "$DATA_DIR/true_user_profile.pkl" --max_his_len "$RECENT_N" --max_len "$MAX_TOKENS" \
    --lora_dir "$SAVE_DIR/best_lora_adapter" --ckpt_path "$SAVE_DIR/best_checkpoint.pt" \
    --output_path "$SAVE_DIR/test_distribution.pkl" --save_jsonl 1
