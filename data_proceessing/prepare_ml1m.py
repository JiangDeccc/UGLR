#!/usr/bin/env python3
"""Prepare raw MovieLens 1M for RPG: disjoint users and chronological text lists."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import random

AGES = {1: 'Under 18', 18: '18-24', 25: '25-34', 35: '35-44',
        45: '45-49', 50: '50-55', 56: '56+'}
OCCUPATIONS = ['other/not specified', 'academic/educator', 'artist',
               'clerical/admin', 'college/grad student', 'customer service',
               'doctor/health care', 'executive/managerial', 'farmer',
               'homemaker', 'K-12 student', 'lawyer', 'programmer', 'retired',
               'sales/marketing', 'scientist', 'self-employed',
               'technician/engineer', 'tradesman/craftsman', 'unemployed', 'writer']


def prepare(source, output, seed=42, history_limit=200):
    if history_limit <= 0:
        raise ValueError('history_limit must be positive')
    profiles = {}
    age_ids = {age: i for i, age in enumerate(AGES)}
    for line in (source / 'users.dat').read_text().splitlines():
        uid, gender, age, occupation, zipcode = line.split('::')
        uid, age, occupation = int(uid), int(age), int(occupation)
        assert uid not in profiles
        profiles[uid] = {
            'label_id': {'gender': {'M': 0, 'F': 1}[gender],
                         'age': age_ids[age], 'occupation': occupation},
            'label_text': {'gender': {'M': 'Male', 'F': 'Female'}[gender],
                           'age': AGES[age], 'occupation': OCCUPATIONS[occupation]},
            'raw': {'gender': gender, 'age': age, 'occupation': occupation,
                    'zip_code': zipcode},
        }
    movies = {}
    for line in (source / 'movies.dat').read_text(encoding='latin-1').splitlines():
        mid, title, genres = line.split('::')
        movies[int(mid)] = (title, genres.replace('|', ', '))
    interactions = defaultdict(list)
    with (source / 'ratings.dat').open() as handle:
        for index, line in enumerate(handle):
            uid, mid, rating, timestamp = map(int, line.strip().split('::'))
            assert uid in profiles and mid in movies and 1 <= rating <= 5
            interactions[uid].append((timestamp, index, mid, rating))
    assert set(interactions) == set(profiles)
    users = sorted(profiles)
    random.Random(seed).shuffle(users)
    n_train, n_val = len(users) * 8 // 10, len(users) // 10
    splits = {'train': users[:n_train], 'val': users[n_train:n_train+n_val],
              'test': users[n_train+n_val:]}
    output.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        with (output / name).open('wb') as handle:
            pickle.dump(value, handle, protocol=4)

    stats = {}
    for split, keys in splits.items():
        histories = {}
        for uid in keys:
            # Stable raw-file order breaks equal-timestamp ties.
            rows = sorted(interactions[uid])[-history_limit:]
            histories[uid] = [
                f'{movies[mid][0]} : Genres({movies[mid][1]}), Rating({rating}/5), '
                f'Rating Time({datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")})'
                for ts, _, mid, rating in rows
            ]
        save(f'{split}_user_history_his{history_limit}.pkl', histories)
        lengths = [len(h) for h in histories.values()]
        stats[split] = {'users': len(keys), 'raw_interactions': sum(len(interactions[k]) for k in keys),
                        'retained_interactions': sum(lengths), 'min': min(lengths),
                        'max': max(lengths), 'mean': sum(lengths)/len(lengths)}
    save('user_splits.pkl', splits)
    save('true_user_profile.pkl', profiles)
    metadata = {
        'dataset': 'ml-1m',
        'source_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                          for name in ('users.dat', 'movies.dat', 'ratings.dat')},
        'total_users': len(users), 'total_movies': len(movies),
        'total_interactions': sum(map(len, interactions.values())),
        'split_method': 'sorted integer user IDs; random.Random(seed).shuffle; 80%/10%/remainder',
        'random_seed': seed, 'splits': {k: len(v) for k, v in splits.items()},
        'history_limit': history_limit, 'history_order': 'timestamp ascending; raw row order for ties',
        'rating_filter': None, 'history_counts': stats,
        'attributes': ['gender', 'age', 'occupation'],
        'label_mapping': {'gender': {'M': 0, 'F': 1},
                          'age_raw_to_id': age_ids,
                          'occupation_id_to_text': dict(enumerate(OCCUPATIONS))},
        'label_counts': {split: {attr: dict(sorted(Counter(profiles[u]['label_id'][attr]
                            for u in keys).items())) for attr in ('gender', 'age', 'occupation')}
                         for split, keys in splits.items()},
        'rpg_format': 'dict[int user_id, list[str]] histories; profiles[user_id][label_id]',
        'missing_attributes': ['education', 'income'],
    }
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'dataset': 'ml-1m', 'splits': metadata['splits'],
                      'history_counts': stats}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--history-limit', type=int, default=200)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir or args.source_dir, args.seed, args.history_limit)
