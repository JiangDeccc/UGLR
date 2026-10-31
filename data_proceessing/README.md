# Raw data to the shared format

Preprocessing for Amazon and MovieLens-1M.

## Output format

- `train_user_history_his200.pkl`, `val_user_history_his200.pkl`, `test_user_history_his200.pkl`: `dict[int, list[str]]`, ordered chronologically and limited to the latest 200 records. The reference Amazon data includes four empty histories (three training, one test), retained for compatibility.
- `true_user_profile.pkl`: `dict[int, {'label_id': dict[str, int], 'label_text': dict[str, str]}]`. MovieLens also includes a `raw` field.
- `user_splits.pkl`: `{'train': [...], 'val': [...], 'test': [...]}` with disjoint users.
- `metadata.json`: input SHA-256 hashes, processing rules, split statistics, and label mappings. Amazon also produces `user2id.pkl`.

Training uses the latest 100 records and an 8192-token limit.

## Amazon

Inputs: `amazon-purchases.csv` and `survey.csv`. Purchase fields: `Survey ResponseID`, `ASIN/ISBN (Product Code)`, `Order Date`, `Title`, `Category`, and `Purchase Price Per Unit`. Survey fields: `Survey ResponseID` and `Q-demos-{gender,age,education,income}`.

1. Apply no user/item core filtering. Sort by `Order Date`, then assign user and item IDs starting at 1 in first-appearance order. Preserve pandas `sort_values` tie ordering for compatibility with the reference histories.
2. Deduplicate item metadata rows, compute mean prices rounded to two decimals, and use the last title/category row for each item.
3. Format each record as `Title : Categories(Category), Price($price), Order Date(timestamp)`.
4. Join survey responses to mapped users. Merge `Some high school or less` into `High school diploma or GED`; merge `55 - 64 years` and `65 and older` into `55 and older`.
5. Retain unknown labels for the training code to handle.

Label IDs:

| Attribute | Mapping |
| --- | --- |
| Gender | Male=0, Female=1, Other=2, Prefer not to say=3 |
| Education | High school or below=0, Bachelor's degree=1, Graduate/professional degree=2, Prefer not to say=3 |
| Age | 18-24=0, 25-34=1, 35-44=2, 45-54=3, 55+=4 |
| Income | <25k=0, 25-49,999=1, 50-74,999=2, 75-99,999=3, 100-149,999=4, 150k+=5, Prefer not to say=6 |

For **exact split reproduction**, supply `--reference-dir` containing the existing `user2id.pkl` and `user_splits.pkl`. The original split algorithm cannot be recovered from the raw CSV alone. The script checks that the reference ID mapping matches the raw data before reusing the splits.

Without `--reference-dir`, the script creates a new 80/10/10 split using sorted integer IDs and `random.Random(seed).shuffle` (default seed: 42). This is not guaranteed to match the reference experiment. Both modes check user coverage and split disjointness.

```bash
python data_proceessing/prepare_amazon.py --source-dir /path/to/amazon \
  --reference-dir /path/to/original_mapping_and_splits --output-dir data/amazon
# Omit --reference-dir to create a new split.
```

## MovieLens-1M

Inputs: `users.dat`, `movies.dat`, and `ratings.dat`, separated by `::`.

1. Retain all ratings. Decode movie metadata using latin-1.
2. Sort original integer user IDs and shuffle with `random.Random(42)`. Allocate 80% to training, 10% to validation, and the remainder to testing.
3. Sort each user's interactions by timestamp, breaking ties by source row order. Keep the latest 200 records.
4. Include title, genres, rating, and UTC timestamp in each history record.
5. Map gender M=0, F=1; map raw age values 1, 18, 25, 35, 45, 50, 56 to IDs 0-6; retain occupation IDs 0-20. Do not synthesize education or income labels.

```bash
python data_proceessing/prepare_ml1m.py --source-dir /path/to/ml-1m --output-dir data/ml1m
```

`--history-limit` defaults to 200. Changing it changes output filenames and requires corresponding training input adjustments. Only load trusted pickle files.
