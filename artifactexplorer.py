import csv
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


def load_from_commit_explorer(db, query, dataset_file, reload_from_commit_explorer, label_func = None) -> List[Dict]:
    file_exists = Path(dataset_file).exists()
    if not reload_from_commit_explorer and not file_exists:
        print(f"File {Path(dataset_file)} does not exists")
    if reload_from_commit_explorer or not file_exists:
        print(f"Loading artifacts for {query} from commit explorer to {dataset_file}")
        commits = db.commits.find(query)
        dataset = []
        no_files_counter = 0
        for commit in tqdm(commits):
            if 'files' not in commit:
                no_files_counter += 1
                continue
            changes = []
            if not isinstance(commit['files'], list):
                continue
            for file in commit['files']:
                file_change = file['change'] if 'change' in file else (file['changes'] if 'changes' in file else '')
                changes.append({'filename': file['filename'], 'changes': file_change})
            dct = {'sha': commit['_id'], 'changes': changes}
            if 'message' in commit:
                dct['message'] = commit['message']
            if label_func is not None:
                dct['label'] = label_func(commit)
            dataset.append(dct)
        print(f'Warning: skipped commits for which the changes were not collected: {no_files_counter}')
        with jsonlines.open(dataset_file, 'w') as writer:
            writer.write_all(dataset)
    else:
        print(f"Loading dataset from file: {dataset_file}")
        with jsonlines.open(dataset_file) as reader:
            dataset = [i for i in reader]
    return dataset


def load_dataset_by_query(query, dataset_file, reload_from_commit_explorer, label_func) -> List[Dict]:
    return load_from_commit_explorer(db, query, dataset_file, reload_from_commit_explorer, label_func)


def load_dataset_by_id(id: str, dataset_file, reload_from_commit_explorer, label_func) -> List[Dict]:
    return load_from_commit_explorer(db, {id: {"$exists":  True}}, dataset_file, reload_from_commit_explorer, label_func)


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
        update = {'$set': {f"bohr.label_model.{model_name}/0_1": {'label': row['prob_CommitLabel.BugFix']}}}
        operations.append(UpdateOne({"_id": row['sha']}, update, upsert=True))

    db.commits.bulk_write(operations)


if __name__ == '__main__':
    model_names = ['gitcproc']
    for model_name in model_names:
        path = f'/Users/hlib/dev/bohr-workdir-bugginess/runs/bugginess/{model_name}/bohr.200k_commits/labeled.csv'
        upload_label_model_labels(path, model_name)
    load_dataset_by_id('bohr.200k_commits', 'datasets/200k_commits.jsonl', False, lambda c: c['bohr']['label_model'])
    # path = '/Users/hlib/dev/diff-classifier/datasets/200k_commits_valid_long.jsonl'
    # operations = []
    # with jsonlines.open(path) as f:
    #     for datapoint in f:
    #         update = {'$set': {f"bohr.large_change": True}}
    #         operations.append(UpdateOne({"_id": datapoint['sha']}, update, upsert=True))
    # db.commits.bulk_write(operations)


