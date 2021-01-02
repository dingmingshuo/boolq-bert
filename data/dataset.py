import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class BoolQDataset(Dataset):
    def __init__(self, input_ids, attention_masks, answers):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.answers = answers

    def __getitem__(self, index):
        ids = self.input_ids[index]
        mask = self.attention_masks[index]
        answer = self.answers[index]
        return ids, mask, answer

    def __len__(self):
        return len(self.answers)


def encode_data(tokenizer, questions, passages, max_length):
    input_ids = []
    attention_masks = []

    for question, passage in tqdm(zip(questions, passages)):
        encoded_data = tokenizer.encode_plus(
            question, passage, truncation=True, padding='max_length', max_length=max_length)
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return input_ids, attention_masks


def get_train_data(data_path, train_data_file, tokenizer, max_seq_length):
    train_data_file = os.path.join(data_path, train_data_file)
    train_data_df = pd.read_json(train_data_file, lines=True, orient='records')

    passages_train = train_data_df.passage.values
    questions_train = train_data_df.question.values
    answers_train = train_data_df.answer.values.astype(int)

    print("Importing train datas from %s:" % train_data_file)
    input_ids_train, attention_masks_train = encode_data(
        tokenizer, questions_train, passages_train, max_seq_length)

    return BoolQDataset(input_ids_train, attention_masks_train, answers_train)


def get_dev_data(data_path, dev_data_file, tokenizer, max_seq_length):
    dev_data_file = os.path.join(data_path, dev_data_file)
    dev_data_df = pd.read_json(dev_data_file, lines=True, orient='records')
    passages_dev = dev_data_df.passage.values
    questions_dev = dev_data_df.question.values
    answers_dev = dev_data_df.answer.values.astype(int)

    print("Importing dev datas from %s:" % dev_data_file)
    input_ids_dev, attention_masks_dev = encode_data(
        tokenizer, questions_dev, passages_dev, max_seq_length)

    return BoolQDataset(input_ids_dev, attention_masks_dev, answers_dev)
