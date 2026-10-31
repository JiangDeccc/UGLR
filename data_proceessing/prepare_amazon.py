"""Amazon raw CSV -> unified RPG format (extracted from the original notebook)."""
import numpy as np
import pandas as pd
import os
import random
import pickle
from collections import defaultdict
import json
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

import pickle
import pandas as pd

gender_map_dict = {
    'Male': 0,
    'Female': 1,
    'Other': 2,
    'Prefer not to say': 3,
}

age_map_dict = {
    '18 - 24 years': 0,
    '25 - 34 years': 1,
    '35 - 44 years': 2,
    '45 - 54 years': 3,
    '55 and older': 4,
}

education_map_dict = {
    'High school diploma or GED': 0,
    "Bachelor's degree": 1,
    'Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS, etc)': 2,
    'Prefer not to say': 3,
}

income_map_dict = {
    'Less than $25,000': 0,
    '$25,000 - $49,999': 1,
    '$50,000 - $74,999': 2,
    '$75,000 - $99,999': 3,
    '$100,000 - $149,999': 4,
    '$150,000 or more': 5,
    'Prefer not to say': 6,
}

def normalize_profile_labels(profile):
    """Normalize raw survey text to the education-3 / age-5 label schema."""
    profile = dict(profile)
    if profile["education"] == "Some high school or less":
        profile["education"] = "High school diploma or GED"
    if profile["age"] in ("55 - 64 years", "65 and older"):
        profile["age"] = "55 and older"
    return profile


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def check_k_core(dataframe, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user in dataframe['uid'].tolist():
        user_count[user] += 1
    for item in dataframe['iid'].tolist():
        item_count[item] += 1
    for user in user_count:
        if user_count[user] < user_core:
            return user_count, item_count, False

    for item in item_count:
        if item_count[item] < item_core:
            return user_count, item_count, False

    return user_count, item_count, True

def delete_users_from_df(dataframe, user_list):
    dataframe = dataframe.drop(dataframe[dataframe['uid'].isin(user_list)].index).reset_index(drop=True)
    return dataframe

def delete_items_from_df(dataframe, item_list):
    dataframe = dataframe.drop(dataframe[dataframe['iid'].isin(item_list)].index).reset_index(drop=True)
    return dataframe

def filter_k_core(dataframe, user_core, item_core):
    user_count, item_count, isKcore = check_k_core(dataframe, user_core, item_core)
    while not isKcore:
        delete_user_list = [user for user in user_count if user_count[user] < user_core]
        delete_item_list = [item for item in item_count if item_count[item] < item_core]
        dataframe = delete_users_from_df(dataframe, delete_user_list)
        dataframe = delete_items_from_df(dataframe, delete_item_list)

        user_count, item_count, isKcore = check_k_core(dataframe, user_core, item_core)
    return dataframe

def load_data(raw_data_folder, raw_data_name, user_core, item_core):
    raw_data_path = os.path.join(raw_data_folder, raw_data_name, 'amazon-purchases.csv')
    print('Loading Amazon purchase data')
    df = pd.read_csv(raw_data_path)
    print(f'apply user {user_core} core and item {item_core} core filters')
    df.rename(columns={'Survey ResponseID': 'uid', 'ASIN/ISBN (Product Code)': 'iid', 'Order Date': 'timestamp'}, inplace=True)
    df = filter_k_core(df, user_core, item_core)
    df = df.sort_values('timestamp')
    users = df['uid'].tolist()
    items = df['iid'].tolist()
    timestamps = df['timestamp'].tolist()
    user_num = len(df['uid'].unique())
    item_num = len(df['iid'].unique())
    print(f'The {raw_data_name} dataset has {len(users)} data with {user_num} users and {item_num} items')
    assert len(users) == len(items)
    user_sequence = defaultdict(list)
    user_sequence_timestamp = defaultdict(list)
    for i in range(len(users)):
        user = str(users[i])
        item = str(items[i])
        user_sequence[user].append(item)
        user_sequence_timestamp[user].append(timestamps[i])
    return user_sequence, user_sequence_timestamp, df

# map user and item to unique index
def id_mapping(filtered_df, user_sequence, user_sequence_timestamp, dataset):
    user_to_index = {user: i+1 for i, user in enumerate(filtered_df['uid'].unique())}
    item_to_index = {item: i+1 for i, item in enumerate(filtered_df['iid'].unique())}
    index_to_user = {i: user for user, i in user_to_index.items()}
    index_to_item = {i: item for item, i in item_to_index.items()}
    # save_pickle(user_to_index, f'{dataset}/user2id.pkl')  # user_to_index.pkl
    # save_pickle(item_to_index, f'{dataset}/item2id.pkl')  # item_to_index.pkl
    # save_pickle(index_to_user, f'{dataset}/id2user.pkl')  # index_to_user.pkl
    # save_pickle(index_to_item, f'{dataset}/id2item.pkl')  # index_to_item.pkl

    filtered_df['uid'] = filtered_df['uid'].apply(lambda x: user_to_index[x])
    filtered_df['iid'] = filtered_df['iid'].apply(lambda x: item_to_index[x])
    # user_sequence mapping
    mapped_user_sequence = {}
    for user, items in user_sequence.items():
        mapped_user = user_to_index[user]
        mapped_items = [item_to_index[item] for item in items if item in item_to_index]
        mapped_user_sequence[mapped_user] = mapped_items
    mapped_user_sequence_timestamp = {}
    for user, timestamps in user_sequence_timestamp.items():
        mapped_user = user_to_index[user]
        mapped_user_sequence_timestamp[mapped_user] = timestamps
    return filtered_df, mapped_user_sequence, mapped_user_sequence_timestamp

def load_user_history(user_data: dict, user_timestamp: dict, user_id: str, item_id: str = '0', max_history_length=None):
    history = user_data[user_id]
    history_timestamp = user_timestamp[user_id]

    if item_id in history:
        idx = history.index(item_id)

        if max_history_length is not None:
            start_idx = max(0, idx - max_history_length + 1)
            return history[start_idx:idx], history_timestamp[start_idx:idx]
        else:
            return history[:idx], history_timestamp[:idx]
    elif item_id == '0':
        if max_history_length is not None:
            start_idx = max(0, len(history) - max_history_length)
            return history[start_idx:], history_timestamp[start_idx:]
        else:
            return history, history_timestamp
    else:
        raise ValueError(f"Item ID {item_id} not found in user history for user {user_id}.")

def generate_history_string(user_ids, user_data, user_timestamp, item_meta, dataset_name, item_id='0', max_history_length=50):
    input_history = dict()
    def generate_his_item(item_id, timestamp):
        if dataset_name == "amazon":
            title = item_meta[int(item_id)]['title']
            categories = item_meta[int(item_id)]['category']
            price = item_meta[int(item_id)]['price']
            # return f"{title}"
            # return f"{title} : Categories({categories})"
            # return f"{title} : Categories({categories}), Price(${price})"
            return f"{title} : Categories({categories}), Price(${price}), Order Date({timestamp})"
        else:
            raise NotImplementedError("Invalid dataset name.")
    for user_id in user_ids:
        history, history_timestamp = load_user_history(user_data, user_timestamp, user_id, item_id=item_id, max_history_length=max_history_length)
        # cands = candidates[int(user_id)][item_id][:candidate_num]
        history_str = [generate_his_item(history[i], history_timestamp[i]) for i in range(len(history))]
        # history_str = '\n'.join(history_str)
        input_history[user_id] = history_str
    return input_history


def prepare(source, output, seed=42, history_limit=200, reference_dir=None):
    from pathlib import Path
    import hashlib
    if history_limit <= 0:
        raise ValueError('history_limit must be positive')
    source, output = Path(source), Path(output)
    sequences, timestamps, interactions = load_data(str(source.parent), source.name, 0, 0)
    generated_user_index = {user: i + 1 for i, user in enumerate(interactions['uid'].unique())}
    interactions, sequences, timestamps = id_mapping(interactions, sequences, timestamps, 'amazon')
    # Match the notebook: mean price over distinct metadata rows, last row wins for title/category.
    item_info = interactions[['iid', 'Title', 'Category', 'Purchase Price Per Unit']].drop_duplicates().reset_index(drop=True)
    item_info['Purchase Price Per Unit'] = item_info.groupby('iid')['Purchase Price Per Unit'].transform('mean').round(2)
    item_info = item_info.drop_duplicates().reset_index(drop=True).fillna('')
    metadata = {row['iid']: {'title': row['Title'], 'category': row['Category'],
                            'price': row['Purchase Price Per Unit']}
                for _, row in item_info.iterrows()}
    if reference_dir is not None:
        reference_dir = Path(reference_dir)
        user_index = load_pickle(reference_dir / 'user2id.pkl')
        if user_index != generated_user_index:
            raise ValueError('Original user2id does not match raw CSV ordering; refusing to mix user identities')
        splits = load_pickle(reference_dir / 'user_splits.pkl')
        split_method = 'reuse reference user_splits.pkl; validate user2id.pkl against raw mapping'
    else:
        user_index = generated_user_index
        users = sorted(sequences)
        random.Random(seed).shuffle(users)
        ntrain, nval = len(users) * 8 // 10, len(users) // 10
        splits = {'train': users[:ntrain], 'val': users[ntrain:ntrain+nval], 'test': users[ntrain+nval:]}
        split_method = 'NEW split: sorted integer IDs; random.Random(seed).shuffle; 80/10/remainder'
    flat = [u for split in ('train', 'val', 'test') for u in splits[split]]
    if len(flat) != len(set(flat)) or set(flat) != set(sequences):
        raise ValueError('Splits must be disjoint and cover all interaction users')
    profiles = {}
    maps = {'gender': gender_map_dict, 'age': age_map_dict,
            'education': education_map_dict, 'income': income_map_dict}
    for _, row in pd.read_csv(source / 'survey.csv').iterrows():
        response_id = row['Survey ResponseID']
        if response_id not in user_index:
            continue
        uid = user_index[response_id]
        if uid in profiles:
            raise ValueError(f'Duplicate survey user: {uid}')
        labels = normalize_profile_labels({attr: row[f'Q-demos-{attr}'] for attr in maps})
        profiles[uid] = {'label_id': {attr: mapping[labels[attr]] for attr, mapping in maps.items()},
                         'label_text': labels}
    if set(flat) - profiles.keys():
        raise ValueError('Some interaction users have no survey labels')
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ('train', 'val', 'test'):
        histories = generate_history_string(splits[split], sequences, timestamps, metadata,
                                            'amazon', max_history_length=history_limit)
        save_pickle(histories, output / f'{split}_user_history_his{history_limit}.pkl')
        counts[split] = {'users': len(histories), 'interactions': sum(map(len, histories.values()))}
    save_pickle(profiles, output / 'true_user_profile.pkl')
    save_pickle(user_index, output / 'user2id.pkl')
    save_pickle(splits, output / 'user_splits.pkl')
    report = {'dataset': 'amazon', 'history_limit': history_limit, 'seed': seed,
              'split_method': split_method, 'counts': counts, 'item_frequency_filter': None,
              'sort': 'pandas sort_values(timestamp), matching original notebook (including tie ordering)',
              'pandas_version': pd.__version__, 'label_mapping': maps,
              'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in ('amazon-purchases.csv', 'survey.csv')}}
    (output / 'metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, help='Original user2id.pkl + user_splits.pkl for exact experiment reproduction')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--history-limit', type=int, default=200)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.seed, args.history_limit, args.reference_dir)
