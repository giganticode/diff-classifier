import csv
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from jsonlines import jsonlines
from pymongo import UpdateOne, MongoClient
from tqdm import tqdm

write_password = '' #add password here if you want to load changes to commit explorer
if write_password:
    db = MongoClient(f'mongodb://write-user:{write_password}@10.10.20.160:27017')['commit_explorer']
else:
    db = MongoClient(f'mongodb://read-only-user:123@10.10.20.160:27017')['commit_explorer']


def load_dataset(id, label_func) -> List[Dict]:
    import dvc.api

    repo = 'https://github.com/giganticode/bohr-workdir-bugginess'
    path = f'cached-datasets/{id}.jsonl'

    dataset = []
    print(f"Loading dataset from file: {repo}/{path}")
    datasets_dir = Path(__file__).parent / 'datasets'
    if not datasets_dir.exists():
        datasets_dir.mkdir()
    full_path = f'{str(datasets_dir)}/{id}.jsonl'
    if not os.path.exists(full_path):
        with dvc.api.open(path, repo=repo, mode='rb') as fd:
            with open(full_path, 'wb') as g:
                g.write(fd.read())
    with jsonlines.open(full_path) as g:
        for commit in tqdm(g):
            commit['label'] = label_func(commit)
            dataset.append(commit)

    return dataset


def upload_transformer_labels(file):
    operations = []
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        _ = next(reader)
        for sha, label, probability, _, _ in reader:
            update = {'$set': {"bohr.change_transformer_label/0_1": {'label': label, 'probability': probability}}}
            operations.append(UpdateOne({"_id": sha}, update, upsert=True))

        db.commits.bulk_write(operations)


def upload_label_model_labels(file, model_name):
    if '.' in model_name:
        raise ValueError(f'Model name cannot contain ".", passed: {model_name}')
    operations = []
    df = pd.read_csv(file)
    for _, row in tqdm(df.iterrows()):
        update = {'$set': {f"bohr.label_model.message_keywords_file_metrics_transformer/0_3": {'label': row['prob_CommitLabel.BugFix']}}}
        operations.append(UpdateOne({"_id": row['sha']}, update, upsert=True))

    db.commits.bulk_write(operations)


if __name__ == '__main__':
    # model_names = ['dataset_debugging']
    # for model_name in model_names:
    #     path = f'/Users/hlib/dev/bohr-workdir-bugginess/runs/bugginess/{model_name}/commits_200k_with_files_no_merges/labeled.csv'
    #     upload_label_model_labels(path, model_name)
    #load_dataset_by_id('bohr.200k_commits', 'datasets/200k_commits.jsonl', False, lambda c: c['bohr']['label_model'])
    # path = '/Users/hlib/dev/diff-classifier/datasets/200k_commits_valid_long.jsonl'
    # operations = []
    # with jsonlines.open(path) as f:
    #     for datapoint in f:
    #         update = {'$set': {f"bohr.large_change": True}}
    #         operations.append(UpdateOne({"_id": datapoint['sha']}, update, upsert=True))
    # db.commits.bulk_write(operations)
    print(load_dataset('levin_files', lambda c: ('BugFix' if c['manual_labels']['levin']['bug'] == 1 else 'NonBugFix'))[0])


