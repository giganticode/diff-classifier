# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from scipy.special import softmax

from artifactexplorer import upload_transformer_labels, load_test_dataset, load_conventional_commit_changes

os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


reload_from_commit_explorer = False


MAX_LENGTH = 512

NEXT_FILE_TOKEN = '<nextfile>'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    output_mode: Optional[str] = field(default="classification")
    eval_test: bool = field(default=False)


def condense_changes(changes: List[Dict]) -> str:
    # very naive way
    return "\n".join([file['filename'] + f' {NEXT_FILE_TOKEN} ' + file['changes'] for file in changes])[:MAX_LENGTH * 10]

def tokenize_dataset(ds, tokenizer, label_map, no_ground_truth):
    changes = []
    labels = []
    total_chars_in_batch = 0
    for datapoint in ds:
        cut_change = condense_changes(datapoint['changes'])
        changes.append(cut_change)
        if not no_ground_truth:
            labels.append(label_map[datapoint['label']])
        total_chars_in_batch += len(cut_change)

    print(f'Tokenizing another chunk of data with length {len(changes)}, '
          f'total chars in batch: {total_chars_in_batch} (average: {float(total_chars_in_batch) / len(changes)}),')

    encoding = tokenizer(
        changes,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    label_ids_dtype = torch.float32 if len(label_map) == 1 else torch.int64
    label_ids_t = torch.tensor(labels, dtype=label_ids_dtype)

    if no_ground_truth:
        return [{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'sha': dp['sha']
        } for attention_mask, input_ids, dp in zip(encoding["attention_mask"], encoding["input_ids"], ds)]
    else:
        return [{
            "label": label,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            'sha': dp['sha']
        } for label, attention_mask, input_ids, dp in zip(label_ids_t, encoding["attention_mask"], encoding["input_ids"], ds)]


class LazyDataset(IterableDataset):
    def __init__(self, ds, tokenizer, label_map, no_ground_truth):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise AssertionError()

        self.ds = ds
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.no_ground_truth = no_ground_truth

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return iter(tokenize_dataset(self.ds, self.tokenizer, self.label_map, self.no_ground_truth))

    @classmethod
    def from_change(cls, change, tokenizer, label_map) -> 'LazyDataset':
        return cls([{'change': change}], tokenizer, label_map, True)


def split_dataset(dataset: List[Dict]) -> Tuple[List, List]:
    train, val = [], []
    for datapoint in dataset:
        if datapoint['sha'][-1] in ['d', 'e', 'f']:
            val.append(datapoint)
        else:
            train.append(datapoint)
    return train, val


def to_chain_of_simple_datasets(dataset, tokenizer, label_map, no_groud_truth=False):
    chunk_len = 2000
    simple_datasets = [LazyDataset(dataset[i * chunk_len:(i + 1) * chunk_len], tokenizer, label_map, no_groud_truth) for i in range((len(dataset) - 1) // chunk_len + 1)]
    return ChainDataset(simple_datasets)


def compute_metrics_fn(p: EvalPrediction, label_names):
    preds = np.argmax(p.predictions, axis=1)
    print(preds)
    print(p.label_ids)

    print(
        classification_report(
            p.label_ids, preds, target_names=label_names, digits=3, labels=list(range(len(label_names)))
        )
    )

    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average="macro")
    return {
        "acc": acc,
        "f1": f1,
    }


def load_model(model_args, training_args, num_labels):

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    special_tokens_dict = {'additional_special_tokens': ['<eq>', '</eq>', '<ins>', '</ins>', '<del>', '</del>', '<re>', '<to>', '</re>', NEXT_FILE_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    print(f'Vocab size: {tokenizer.vocab_size}')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def create_or_load_label_map(dataset: Optional[List[Dict]], path: str) -> Dict[str, int]:
    if (Path(path) / f"label_map.txt").exists():
        with (Path(path) / f"label_map.txt").open() as f:
            return json.load(f)
    label_dct = {}
    for datapoint in dataset:
        if datapoint['label'] not in label_dct:
            label_dct[datapoint['label']] = 0
        label_dct[datapoint['label']] += 1

    print(label_dct)
    label_names = sorted(label_dct.keys())
    label_map = {l: i for i, l in enumerate(label_names)}
    with open(path, 'w') as f:
        json.dump(label_map, f)
    return label_map


def predict(trainer, dataset, ids_to_labels, save_to):
    logging.info("*** Test ***")

    predictions = trainer.predict(test_dataset=dataset).predictions
    predicitions_softmax = softmax(predictions, axis=1)
    labels = np.argmax(predictions, axis=1)
    probabilities = np.max(predicitions_softmax, axis=1)

    if trainer.is_world_process_zero():
        with open(save_to, "w") as writer:
            logger.info("***** Test results *****")
            writer.write("index,prediction, probability\n")
            for datapoint, label, probability in zip(dataset, labels, probabilities):
                writer.write("%s,%s,%f\n" % (datapoint['sha'], ids_to_labels[label], probability))



def main(dataset_id):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(training_args)

    dataset = load_conventional_commit_changes('datasets/conventional_commits_changes.jsonl', reload_from_commit_explorer)
    train_set, valid_set = split_dataset(dataset)
    print(f'Train dataset - {len(train_set)} datapoints')
    print(f'Valid dataset - {len(valid_set)} datapoints')

    test_dataset = load_test_dataset(dataset_id, f'datasets/dataset_{dataset_id}.jsonl', reload_from_commit_explorer, None)

    print(f'Test dataset - {len(test_dataset)} datapoints')

    label_map = create_or_load_label_map(dataset, os.path.join(training_args.output_dir, f"label_map.txt"))
    ids_to_labels = {i: l for l, i in label_map.items()}
    label_names = sorted(label_map.keys())
    num_labels = len(label_map)

    model, tokenizer = load_model(model_args, training_args, num_labels)

    # if model_args.eval_test:
    #     print("**** TEST EVAL *****")
    #     eval_dataset = to_chain_of_simple_datasets(valid_set, tokenizer, label_map)
    #     train_dataset = None
    # else:
    #     print("**** TRAINING ******")
    #     train_dataset = to_chain_of_simple_datasets(train_set, tokenizer, label_map)
    #     print("Tokenization example")
    #     first_elm = next(iter(train_dataset))
    #     print([tokenizer.decode(s) for s in first_elm['input_ids'].tolist()])
    #     print(first_elm['input_ids'].shape)
    #     print(first_elm['input_ids'].shape)
    #
    #     eval_dataset = to_chain_of_simple_datasets(valid_set, tokenizer, label_map)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=lambda p: compute_metrics_fn(p, label_names),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [valid_set]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = lambda p: compute_metrics_fn(p, label_names)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        save_to = os.path.join(training_args.output_dir, f"assigned_labels_{dataset_id}.csv")
        test_dataset = to_chain_of_simple_datasets(test_dataset, tokenizer, label_map, True)
        predict(trainer, test_dataset, ids_to_labels, save_to)
        upload_transformer_labels(save_to)
    return eval_results


if __name__ == "__main__":
    import sys
    sys.argv.extend(['--model_name_or_path', "bohr_model"])
    sys.argv.extend(['--output_dir', 'bohr_model'])
    sys.argv.extend(['--per_device_eval_batch_size', '32'])
    sys.argv.extend(['--do_predict'])
    main('manual_labels.berger')
