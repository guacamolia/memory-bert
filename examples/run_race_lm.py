""" run local BERT language model """

import argparse
import glob
import json
import logging
import os
import sys
from itertools import repeat
# from multiprocessing import Pool
import multiprocessing as mp

from torch.multiprocessing import Pool

import torch
import wandb
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

sys.path.insert(0, "/home/okovaleva/projects/memory-bert/src")
from transformers import (
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from run_language_modeling import mask_tokens
from utils_multiple_choice import RaceProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RaceDatasetLM(Dataset):
    def __init__(self, example, tokenizer, block_size=128):
        self.text = example.contexts[0]
        self.examples = []
        self.dataset_id = example.example_id
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.text))

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        for i in range(0, max(len(tokenized_text) - block_size + 1, 1), block_size):  # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def train_local_lm(train_dataset, model, tokenizer, args, prefix=""):

    if 'high' in  train_dataset.dataset_id:
        set_prefix = 'h'
    else:
        set_prefix = 'm'
    dataset_id = set_prefix + train_dataset.dataset_id.split('/')[-1][:-4]
    wandb.init(project=f"bert-memory-race-full-{prefix}", reinit=True, name=dataset_id)
    output_dir = os.path.join(args.output_dir, f"race_{prefix.split('_')[1]}")
    output_dir = os.path.join(output_dir, "{}".format(dataset_id))
    os.makedirs(output_dir, exist_ok=True)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    batch_size = torch.cuda.device_count() * args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate)

    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            # "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    epochs_trained = 0
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")

    model_to_save = (
        model.module if hasattr(model, "module") else model
    )

    # Evaluate pre-trained model first
    model.eval()
    tr_loss = 0
    epoch_correct = 0
    epoch_total = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        inputs, labels = mask_tokens(batch, tokenizer, args)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, masked_lm_labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        loss = loss.mean()
        tr_loss += loss.item()

        batch_correct, batch_total = count_correct(labels, outputs[1])
        epoch_correct += batch_correct
        epoch_total += batch_total

    epoch_loss = tr_loss / (step + 1)
    if epoch_total != 0:
        accuracy = epoch_correct / epoch_total * 100
    else:
        accuracy = None
    perplexity = torch.exp(torch.tensor(epoch_loss))
    wandb.log({
        'training_loss': epoch_loss,
        'training_perplexity': perplexity,
        'tok_rec_accuracy': accuracy
    }, step=global_step)

    # Run training
    max_moving_acc = 0
    model.train()
    window = 5
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        tr_loss = 0
        epoch_correct = 0
        epoch_total = 0
        accuracies = [None] * window  # window of 5 for moving average
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens(batch, tokenizer, args)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, masked_lm_labels=labels)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()
            loss.backward()

            tr_loss += loss.item()

            batch_correct, batch_total = count_correct(labels, outputs[1])
            epoch_correct += batch_correct
            epoch_total += batch_total

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

        epoch_loss = tr_loss / (step + 1)
        if epoch_total != 0:
            accuracy = epoch_correct / epoch_total * 100
        else:
            accuracy = None
        accuracies.append(accuracy)
        accuracies.pop(0)

        # check for NaNs when calculating the moving average
        values = [value for value in accuracies if value is not None]
        if values:
            moving_accuracy = sum(values) / len(values)
        else:
            moving_accuracy = 0
        perplexity = torch.exp(torch.tensor(epoch_loss))
        if moving_accuracy > max_moving_acc:
            model_to_save.save_pretrained(output_dir) # save if the average over the past 5 epochs was the best
            save_results(output_dir, {'accuracy': accuracy, 'loss': epoch_loss, 'perplexity': perplexity.item()})
            max_moving_acc = moving_accuracy
        wandb.log({
                'training_loss': epoch_loss,
                'training_perplexity': perplexity,
                'tok_rec_accuracy': accuracy
            }, step=global_step)

    wandb.join()

    return global_step, epoch_loss


def count_correct(labels, outputs):
    """ Count correctly reconstructed tokens (out of masked tokens) """
    predictions = torch.argmax(outputs, dim=-1)
    mask = labels != -100
    correct = torch.eq(predictions, labels)[mask].sum().item()
    total = mask.sum().item()
    return correct, total


def save_results(location, results):
    output_path = os.path.join(location, 'results.json')
    with open(output_path, 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--n_gpu", default=1, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    args = parser.parse_args()

    # Initialize the model
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer

    config = config_class.from_pretrained(os.path.join(args.model_name_or_path, "bert_config.json"))
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=False, config=config)

    # Transferring to CUDA
    model.to(device)
    model = torch.nn.DataParallel(model)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Initialize the data
    processor = RaceProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    # dev_examples = processor.get_dev_examples(args.data_dir)
    # test_examples = processor.get_test_examples(args.data_dir)

    # all_examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}
    # logger.info(f"Total of {len(train_examples) + len(dev_examples) + len(test_examples)} found.")

    # Trying parallelism
    trained_so_far_1 = glob.glob(os.path.join("/data3/okovaleva/bert-memory", "race_train/*"))
    trained_ids_1 = [item.split('/')[-1] for item in trained_so_far_1]
    trained_so_far_2 = glob.glob(os.path.join(args.output_dir, f"race_train/*"))
    trained_ids_2 = [item.split('/')[-1] for item in trained_so_far_2]
    trained_ids_all = trained_ids_1 + trained_ids_2

    to_train = []
    for example in train_examples:
        if 'high' in example.example_id:
            prefix = 'h'
        else:
            prefix = 'm'
        example_id = prefix + example.example_id.split('/')[-1][:-4]
        if example_id not in trained_ids_all:
            to_train.append(example)

    train_datasets = []
    for example in tqdm(to_train):
        train_datasets.append(RaceDatasetLM(example, tokenizer))
    train_datasets = [dataset for dataset in train_datasets if len(dataset) > 0]
    prefixes = [f"small_train_multiprocessing_{str((i+1)//250)}" for i in range(len(train_datasets))]

    mp = mp.get_context('spawn')
    pool = mp.Pool(processes=5)
    pool.starmap(train_local_lm, zip(train_datasets, repeat(model), repeat(tokenizer), repeat(args), prefixes))
    pool.close()
    pool.join()



