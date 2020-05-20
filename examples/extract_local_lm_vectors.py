import argparse
import os
import pickle
import torch
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm

from transformers import (
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    get_linear_schedule_with_warmup
)

from multiple_choice.utils_multiple_choice import RaceProcessor, InputFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RaceDatasetLM(Dataset):
    def __init__(self, example, tokenizer, block_size=128):
        self.text = example.contexts[0] #CHANGED
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
        "--max_length",
        default=128,
        type=str,
        required=False,
        help="Maximum sequence length for encoding the question + the answer>",
    )

    args = parser.parse_args()

    # intialize the classes for the model config and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
    tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    # Initialize the data
    processor = RaceProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    all_examples = {'train': train_examples, 'dev': dev_examples, 'test': test_examples}

    # extract vectors
    vectors = defaultdict(list)
    for set, examples in all_examples.items():
        if set == 'test':
            model_dir = '/data2/okovaleva/bert-memory-models/small/race_test/'
        elif set == 'dev':
            model_dir = '/data2/okovaleva/bert-memory-models/small/race_dev/'
        else:
            model_dir = '/data2/okovaleva/bert-memory-models/small/race_train/'

        for example in tqdm(examples, desc=set):
            example_id = example.example_id.split('/')[-1][:-4]
            if 'high' in example.example_id:
                example_id = 'h' + example_id
            else:
                example_id = 'm' + example_id
            cur_dir = os.path.join(model_dir, example_id)
            if not os.path.exists(cur_dir):
                cur_dir = os.path.join('/data3/okovaleva/bert-memory/race_train/', example_id)
            config = config_class.from_pretrained(os.path.join(cur_dir, "config.json"))
            model = model_class.from_pretrained(cur_dir, from_tf=False, config=config)

            model.to(device)
            # model = torch.nn.DataParallel(model)

            label = example.label
            representations = []
            input_ids = []
            attention_mask = []
            token_type_ids = []
            for i in range(len(example.endings)):
                text_a = example.question
                text_b = example.endings[i]
                inputs = tokenizer.encode_plus(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=args.max_length,
                    pad_to_max_length=True,
                    return_overflowing_tokens=True,
                )

                input_ids.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
                attention_mask.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))
                token_type_ids.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))

            input_ids = torch.stack(input_ids, dim=0).to(device)
            attention_mask = torch.stack(attention_mask, dim=0).to(device)
            token_type_ids = torch.stack(token_type_ids, dim=0).to(device)

            outputs, _ = model.bert(input_ids, attention_mask, token_type_ids)
            cls = outputs[:, 0, :].cpu().detach()

            # populate the vectors dict
            # order of questions is preserved but need to double-check
            vectors[example_id].append(
                {
                    'vectors': cls,
                    'label': label
                }
            )

    with open('/home/okovaleva/projects/memory-bert/trained_lms/pickled/race_vectors.pkl', 'wb') as file:
        pickle.dump(vectors, file)
