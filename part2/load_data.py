import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

# ADDED: extended SQL tokenizer
from tokenizer_utils import build_tokenizer

PAD_IDX = 0

# ADDED: Tokenizer checkpoint and sequence length limits
MODEL_NAME = "google-t5/t5-small"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512

class T5Dataset(Dataset):

    # MODIFIED: Accept an optional shared tokenizer
    def __init__(self, data_folder, split, tokenizer=None):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")

        if tokenizer is None:
            tokenizer = build_tokenizer()

        self.split = split

        self.tokenizer = tokenizer
        self.items = self.process_data(data_folder, split, tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        questions = load_lines(
            os.path.join(data_folder, f"{split}.nl")
        )
        queries = None
        # Only train and dev have ground-truth SQL.
        if split != "test":
            queries = load_lines(
                os.path.join(data_folder, f"{split}.sql")
            )
            if len(questions) != len(queries):
                raise ValueError(f"Question and SQL counts do not match for {split}")

        items = []
        for index, question in enumerate(questions):
            # No task prefix or schema is added.
            encoded = tokenizer(
                question.strip(),
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SOURCE_LEN,
                padding=False,
                return_attention_mask=True,
            )

            item = {
                "encoder_ids": torch.tensor(
                    encoded["input_ids"],
                    dtype=torch.long,
                ),
                "encoder_mask": torch.tensor(
                    encoded["attention_mask"],
                    dtype=torch.long,
                ),
            }

            if queries is not None:
                sql = re.sub(r"\s+", " ", queries[index].strip())
                target_ids = tokenizer.encode(
                    sql,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=MAX_TARGET_LEN,
                )
                item["decoder_targets"] = torch.tensor(
                    target_ids,
                    dtype=torch.long,
                )
            items.append(item)

        return items

    def __len__(self):
        # TODO
        return len(self.items)

    def __getitem__(self, idx):
        # TODO
        return self.items[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = pad_sequence(
        [item["encoder_ids"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = pad_sequence(
        [item["encoder_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    # The provided train_epoch excludes target positions equal to 0.
    decoder_targets = pad_sequence(
        [item["decoder_targets"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    # T5 uses PAD as its decoder start token.
    # Shift targets right:
    # targets: [A, B, EOS]
    # inputs:  [PAD, A, B]
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]
    initial_decoder_inputs = decoder_inputs[:, :1].clone()

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = pad_sequence(
        [item["encoder_ids"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = pad_sequence(
        [item["encoder_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    initial_decoder_inputs = torch.full(
        (len(batch), 1),
        PAD_IDX,
        dtype=torch.long,
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs

# MODIFIED: Accept a shared tokenizer
def get_dataloader(batch_size, split, tokenizer=None):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, tokenizer)  # MODIFIED: Forward the shared tokenizer.
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

# MODIFIED BEGIN: Use one tokenizer for train, dev, and test
def load_t5_data(batch_size, test_batch_size, tokenizer=None):
    if tokenizer is None:
        tokenizer = build_tokenizer()

    train_loader = get_dataloader(batch_size, "train", tokenizer)
    dev_loader = get_dataloader(test_batch_size, "dev", tokenizer)
    test_loader = get_dataloader(test_batch_size, "test", tokenizer)
    
    return train_loader, dev_loader, test_loader
# MODIFIED END


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(
        os.path.join(data_folder, "train.nl")
    )
    train_y = [
        re.sub(r"\s+", " ", sql)
        for sql in load_lines(os.path.join(data_folder, "train.sql"))
    ]
    dev_x = load_lines(
        os.path.join(data_folder, "dev.nl")
    )
    dev_y = [
        re.sub(r"\s+", " ", sql)
        for sql in load_lines(os.path.join(data_folder, "dev.sql"))
    ]
    test_x = load_lines(
        os.path.join(data_folder, "test.nl")
    )
    return train_x, train_y, dev_x, dev_y, test_x
