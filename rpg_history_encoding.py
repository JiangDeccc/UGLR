"""Preserve instructions and the newest complete history records within a token budget."""

HISTORY_PREFIX = (
    "Task: Infer the user's demographic attributes based on the interaction history. "
    "Predict the user's age, gender, education, and income.\n\n"
    "Here is the user's interaction history:\n"
)


def encode_recent_history(tokenizer, history_items, max_his_len, max_len,
                          pad_to_max_length=False, return_offsets_mapping=False):
    """Drop oldest whole records until the exact encoded prompt fits.

    Input records must be ordered oldest to newest. Special tokens count toward
    the budget. An oversized newest record leaves an instruction-only prompt;
    a budget too small for the instructions raises an error.
    """
    if max_his_len <= 0 or max_len <= 0:
        raise ValueError("max_his_len and max_len must be positive")
    items = list(history_items[-max_his_len:])
    for start in range(len(items) + 1):
        retained = items[start:]
        prompt = HISTORY_PREFIX + "\n".join(retained) + "\n"
        encoding = tokenizer(
            prompt, truncation=False, padding=False, max_length=max_len, return_tensors="pt",
            **({"return_offsets_mapping": True} if return_offsets_mapping else {}),
        )
        if encoding["input_ids"].shape[-1] <= max_len:
            if pad_to_max_length:
                encoding = tokenizer(
                    prompt, truncation=False, padding="max_length",
                    max_length=max_len, return_tensors="pt",
                    **({"return_offsets_mapping": True} if return_offsets_mapping else {}),
                )
            return retained, prompt, encoding
    raise ValueError(f"max_len={max_len} is too small for the task instructions and special tokens")
