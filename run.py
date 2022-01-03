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
import logging
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Type, TypeVar, Callable, Union

from scipy.special import softmax

from artifactexplorer import load_dataset

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
    set_seed, PreTrainedModel, PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


reload_from_commit_explorer = False


MAX_LENGTH = 512

NEXT_FILE_TOKEN = '<nextfile>'


class LabelSource:
    def __init__(self, label_map):
        self.label_map = label_map
        self.ids_to_labels = {i: l for l, i in self.label_map.items()}
        self.label_names = sorted(self.label_map.keys())
        self.num_labels = len(label_map)

    def get_label_id_from_datapoint(self, datapoint) -> int:
        return self.label_map[datapoint['label']]

    def get_label_from_id(self, id: int) -> str:
        return self.ids_to_labels[id]


class LabelModelSource(LabelSource):
    def __init__(self, label_map, label_model_name):
        super(LabelModelSource, self).__init__(label_map)
        self.label_model_name = label_model_name

    def get_label_id_from_datapoint(self, datapoint) -> int:
        return round(datapoint['label'][self.label_model_name]['label'])


LabelSourceType = TypeVar('LabelSourceType', bound=LabelSource)


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


class LazyDataset(IterableDataset):
    def __init__(self, ds: List[Dict], tokenizer: PreTrainedTokenizer, label_source: LabelSourceType, no_ground_truth: bool, bimodal: bool = False):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise AssertionError()

        self.ds: List[Dict] = ds
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.label_source: LabelSourceType = label_source
        self.no_ground_truth: bool = no_ground_truth
        self.bimodal = bimodal

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return iter(self.numericalize(self.ds))

    @classmethod
    def from_change(cls, change, tokenizer, label_map) -> 'LazyDataset':
        return cls([{'change': change}], tokenizer, label_map, True)

    @abstractmethod
    def preprocess_input(self, datapoint: Dict) -> str:
        pass

    def numericalize(self, ds):
        text_list = []
        text_pair_list = []
        labels = []
        for datapoint in ds:
            if self.bimodal:
                text, text_pair = self.preprocess_input(datapoint)
                text_pair_list.append(text_pair)
            else:
                text = self.preprocess_input(datapoint)

            text_list.append(text)
            if not self.no_ground_truth:
                labels.append(self.label_source.get_label_id_from_datapoint(datapoint))

        encoding = self.tokenizer(
            text=text_list,
            text_pair=text_pair_list if self.bimodal else None,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        res = []
        for ind, (attention_mask, input_ids, dp) in enumerate(zip(encoding["attention_mask"], encoding["input_ids"], ds)):
            dct = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                'sha': dp['_id'],
                'is_truncated': 'True' if (not (attention_mask == 0).any() ) else 'False', #HACK has to be string so that its ignored by the model
            }
            if not self.no_ground_truth:
                dct['label'] = labels[ind]
            res.append(dct)
        return res


LazyDatasetType = TypeVar('LazyDatasetType', bound=LazyDataset)


class ChangeDataset(LazyDataset):
    def __init__(self, ds: List[Dict], tokenizer: PreTrainedTokenizer, label_source: LabelSourceType, no_ground_truth: bool):
        super(ChangeDataset, self).__init__(ds, tokenizer, label_source, no_ground_truth)

    def preprocess_input(self, datapoint: Dict) -> Union[str, Tuple[str, str]]:
        # very naive way
        return "\n".join([file['filename'] + f' {NEXT_FILE_TOKEN} ' + (file['changes'] if 'changes' in file else '') for file in datapoint['files']])[:MAX_LENGTH * 10]


class MessageDataset(LazyDataset):
    def __init__(self, ds: List[Dict], tokenizer: PreTrainedTokenizer, label_source: LabelSourceType, no_ground_truth: bool):
        super(MessageDataset, self).__init__(ds, tokenizer, label_source, no_ground_truth)

    def preprocess_input(self, datapoint: Dict) -> Union[str, Tuple[str, str]]:
        message = datapoint['message']
        if not isinstance(message, str):
            logger.warning(f'Strange message encountered: {message}')
            message = str(message)
        return message


class MessageChangeDataset(LazyDataset):
    def __init__(self, ds: List[Dict], tokenizer: PreTrainedTokenizer, label_source: LabelSourceType, no_ground_truth: bool):
        super(MessageChangeDataset, self).__init__(ds, tokenizer, label_source, no_ground_truth, True)

    def preprocess_input(self, datapoint: Dict) -> Union[str, Tuple[str, str]]:
        message = datapoint['message']
        if not isinstance(message, str):
            logger.warning(f'Strange message encountered: {message}')
            message = str(message)

        # very naive way
        change = "\n".join([file['filename'] + f' {NEXT_FILE_TOKEN} ' + file['changes'] for file in datapoint['changes']])[:MAX_LENGTH * 10]
        return message, change


@dataclass
class Task:
    name: str
    dataset: str
    dataset_class: Type[LazyDatasetType]
    label_source: LabelSourceType
    test_label_source: LabelSourceType

    def get_pretrained_checkpoint(self) -> str:
        if self.dataset_class.__name__ == 'ChangeDataset':
            return "huggingface/CodeBERTa-small-v1"
        elif self.dataset_class.__name__ == 'MessageDataset':
            return "giganticode/StackOBERTflow-comments-small-v1"
        elif self.dataset_class.__name__ == 'MessageChangeDataset':
            return "microsoft/codebert-base"
        else:
            raise AssertionError()


def split_dataset(dataset: List[Dict]) -> Tuple[List, List]:
    train, val = [], []
    for datapoint in dataset:
        if datapoint['_id'][-1] in ['d', 'e', 'f']:
            val.append(datapoint)
        else:
            train.append(datapoint)
    return train, val


def to_chain_of_simple_datasets(dataset: List[Dict], cls: Type[LazyDatasetType], tokenizer: PreTrainedTokenizer, label_source: LabelSourceType, no_ground_truth: bool=False) -> IterableDataset:
    chunk_len = 2000
    simple_datasets = [cls(dataset[i * chunk_len:(i + 1) * chunk_len], tokenizer, label_source, no_ground_truth) for i in range((len(dataset) - 1) // chunk_len + 1)]
    return ChainDataset(simple_datasets)


def compute_metrics_fn(p: EvalPrediction, label_names: List[str]):
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


def load_model(model_args, training_args, num_labels: int) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

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


def assign_labels(trainer: Trainer, dataset: IterableDataset, label_source: LabelSourceType, test_label_source: LabelSourceType, save_to: str) -> None:
    logging.info("*** Test ***")

    predictions = trainer.predict(test_dataset=dataset).predictions
    predicitions_softmax = softmax(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    probabilities = np.max(predicitions_softmax, axis=1)

    if trainer.is_world_process_zero():
        with open(save_to, "w") as writer:
            logger.info("***** Test results *****")
            writer.write("index,prediction,probability,true_label,truncated\n")
            for datapoint, predicted_label, probability in zip(dataset, predicted_labels, probabilities):
                writer.write("%s,%s,%f,%s,%s\n" % (datapoint['sha'], label_source.get_label_from_id(predicted_label), probability, test_label_source.get_label_from_id(datapoint['label']), datapoint['is_truncated']))


def calc_metrics(trainer: Trainer, tokenized_set: IterableDataset, output_path: str) -> None:
    logger.info("*** Evaluate ***")
    eval_result = trainer.evaluate(eval_dataset=tokenized_set)
    if trainer.is_world_process_zero():
        with open(output_path, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))


def show_tokenization_example(dataset, tokenizer: PreTrainedTokenizer):
    print("Tokenization example")
    first_elm = next(iter(dataset))
    print([tokenizer.decode(s) for s in first_elm['input_ids'].tolist()])
    print(first_elm['input_ids'].shape)
    print(first_elm['input_ids'].shape)


def main(task: Task):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(training_args)

    train_set, valid_set = split_dataset(load_dataset(task.dataset, datasets[task.dataset]))
    print(f'Train dataset - {len(train_set)} datapoints')
    print(f'Valid dataset - {len(valid_set)} datapoints')

    model, tokenizer = load_model(model_args, training_args, task.label_source.num_labels)

    print("**** TRAINING ******")
    tokenized_train_set = to_chain_of_simple_datasets(train_set, task.dataset_class, tokenizer, task.label_source)
    tokenized_valid_set = to_chain_of_simple_datasets(valid_set, task.dataset_class, tokenizer, task.label_source)
    show_tokenization_example(tokenized_train_set, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_valid_set,
        compute_metrics=lambda p: compute_metrics_fn(p, task.label_source.label_names),
    )

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

    if training_args.do_eval and training_args.do_predict:
        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results.txt"
        )
        calc_metrics(trainer, tokenized_valid_set, output_eval_file)
    if training_args.do_predict:
        assign_labels_to_all_datasets(trainer, tokenizer, task, training_args.output_dir)


datasets = {
    'commits_200k_files': lambda c: c['bohr']['label_model'],
    'commits_200k_files_no_merges': lambda c: c['bohr']['label_model'],
    # 'conventional': (lambda: load_dataset_by_query({'conventional_commit/0_1.conventional': True, 'files': {"$exists": True}}, 'datasets/conventional_commits_changes.jsonl', reload_from_commit_explorer, lambda c: c['conventional_commit/0_1']['type'].lower())),
    'levin_files': lambda c: ('BugFix' if c['manual_labels']['levin']['bug'] == 1 else 'NonBugFix'),
    'berger_files': lambda c: ('BugFix' if c['manual_labels']['berger']['bug'] == 1 else 'NonBugFix'),
    'herzig': lambda c: ('BugFix' if c['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else 'NonBugFix'),
    'mauczka_files': lambda c: ('BugFix' if c['manual_labels']['mauczka']['hl_corrective'] == 1 else 'NonBugFix'),
}


TEST_DATASET_NAMES = [
    'levin_files',
    'berger_files',
    'herzig',
    'mauczka_files',
]


def assign_labels_to_all_datasets(trainer, tokenizer, task, output_dir) -> None:
    for test_dataset_name in TEST_DATASET_NAMES:
        test_set = datasets[test_dataset_name]()
        print(f'Test dataset ({test_dataset_name}) - {len(test_set)} datapoints')
        save_to = os.path.join(output_dir, f"assigned_labels_{test_dataset_name}.csv")
        tokenized_test_set = to_chain_of_simple_datasets(test_set, task.dataset_class, tokenizer, task.test_label_source)
        assign_labels(trainer, tokenized_test_set, task.label_source, task.test_label_source, save_to)
        if task.label_source.label_map == task.test_label_source.label_map:
            calc_metrics(trainer, tokenized_test_set, os.path.join(output_dir, f"eval_results_{test_dataset_name}.txt"))
        #upload_transformer_labels(save_to)


tasks = {
    # 'task0': Task("bohr_model", datasets["conventional"], ChangeDataset, LabelSource({"build": 0, "chore": 1, "ci": 2, "docs": 3, "feat": 4, "fix": 5, "perf": 6, "refactor": 7, "style": 8, "test": 9}), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task1': Task("only_message_keywords_message_and_change", "commits_200k_files", MessageChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "only_message_keywords/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task2': Task("all_heuristics_without_issues_message_and_change", "commits_200k_files", MessageChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "message_keywords_file_metrics_transformer/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task3': Task("all_heuristics_with_issues_message_and_change", "commits_200k_files", MessageChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "all_keywords_transformer_filemetrics/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task4': Task("all_heuristics_with_issues_only_message", "commits_200k_files", MessageDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "all_keywords_transformer_filemetrics/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task5': Task("all_heuristics_with_issues_only_change", "commits_200k_files", ChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "all_keywords_transformer_filemetrics/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task7': Task("only_message_keywords_only_message", "commits_200k_files", MessageDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "only_message_keywords/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task8': Task("all_heuristics_without_issues_only_message", "commits_200k_files", MessageDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "message_keywords_file_metrics_transformer/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task9': Task("only_message_keywords_only_change", "commits_200k_files", ChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "only_message_keywords/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task10': Task("gitcproc_only_message", "commits_200k_files", MessageDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "gitcproc/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task11': Task("gitcproc_only_change", "commits_200k_files", ChangeDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "gitcproc/0_1"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
    'task12': Task("only_message_keywords_no_merge_only_message", "commits_200k_files_no_merges", MessageDataset, LabelModelSource({"BugFix": 1, "NonBugFix": 0}, "only_message_keyword/0_2"), LabelSource({"BugFix": 1, "NonBugFix": 0})),
}


def add_common_config(task: Task) -> None:
    sys.argv.extend(['--output_dir', task.name])
    sys.argv.extend(['--per_device_eval_batch_size', '7'])
    sys.argv.extend(['--do_predict'])
    sys.argv.extend(['--overwrite_output_dir'])
    sys.argv.extend(['--per_device_train_batch_size', '7'])
    sys.argv.extend(['--save_steps', '4000'])
    sys.argv.extend(['--num_train_epochs', '3'])
    sys.argv.extend(['--logging_steps', '4000'])
    sys.argv.extend(['--eval_steps', '4000'])
    sys.argv.extend(['--evaluation_strategy', 'steps'])
    sys.argv.extend(['--load_best_model_at_end'])


def training_config(task: Task) -> None:
    add_common_config(task)
    sys.argv.extend(['--model_name_or_path', task.get_pretrained_checkpoint()])
    sys.argv.extend(['--do_train'])
    sys.argv.extend(['--do_eval'])


def evaluation_config(task: Task) -> None:
    add_common_config(task)
    sys.argv.extend(['--model_name_or_path', task.name])


if __name__ == "__main__":
    task = tasks['task5']
    evaluation_config(task)
    main(task)
