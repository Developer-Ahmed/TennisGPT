import numpy as np
import pandas as pd
from datasets import load_dataset
import tiktoken
import os
import math

import torch
from torch.utils.data.dataloader import DataLoader

# initialize hyperparamaters
from config_example import ConfigClass as config

# tokenizer from tiktoken
# for models with bigger and more complex inputs, use newer tokenizers (such as gpt-4)
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

# modify gpt2 tokenizer to add special tokens
special_tokens = {"<|pad|>": 50258}
if not config.pretraining:
    special_tokens["<|user|>"] = 50259
    special_tokens["<|assistant|>"] = 50260

from tiktoken_ext.openai_public import gpt2

# Load the GPT-2 encoding with required parameters from openai_public.py
tokenizer = tiktoken.Encoding(
    name="gpt2_with_pad",  # A custom name 
    pat_str=gpt2()["pat_str"],  
    mergeable_ranks=gpt2()["mergeable_ranks"],  
    special_tokens={
        **gpt2()["special_tokens"],  
        **special_tokens,  # Additional special tokens
    }
)

# create a folder for the dataset if it doesn't exist
dataset_path = f"./data/{config.dataset_name}"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# create train and validation files if they don't exist
if not os.path.exists(f"{dataset_path}/train.bin") and not os.path.exists(f"{dataset_path}/test.bin"):
    if config.huggingface_dataset:
        ds = load_dataset(config.dataset_name)

        def f(txt):
            if len(txt["text"]) == 0:
                td = []
            else:
                td = tokenizer.encode_ordinary(txt["text"]) + [tokenizer.eot_token]
            return {"tokenized_text": td, "num_tokens": len(td)}


        md = ds.map(f, remove_columns=["text"])
        fmd = md.filter(lambda element: len(element["tokenized_text"]) > 0)
        sfmd = fmd["train"].train_test_split(test_size=0.1, seed=2024)

        # load to memory mapped file
        for split in sfmd:
            total_num_tokens = np.sum(sfmd[split]["num_tokens"], dtype="uint64")
            file = np.memmap(f"{dataset_path}/{split}.bin", dtype=np.uint16, mode="w+", shape=(total_num_tokens, ))
            num_batches = config.loading_batches
            loaded_tokens = 0
            n = 0
            for batch_id in range(num_batches):
                current_batch = sfmd[split].shard(num_shards=num_batches, index=batch_id, contiguous=True)
                conc_batch = np.concatenate(current_batch["tokenized_text"])
                file[loaded_tokens:loaded_tokens + len(conc_batch)] = conc_batch
                loaded_tokens += len(conc_batch)
                n += 1

            file.flush()
    else:
        # csv file is assumed
        file_reader = pd.read_csv(f"{config.dataset_location}/{config.dataset_name}.csv", chunksize=1000)

        # tokenize
        all_tokens = np.empty((0,), dtype=np.uint16)
        for chunk in file_reader:
            for pair_id in range(len(chunk)):
                pair = chunk.iloc[pair_id]
                text = "<|user|>" + pair["Question"] + "<|assistant|>" + pair["Answer"]
                tokenized = tokenizer.encode(text, allowed_special="all") 
                tokenized = np.array(tokenized, dtype=np.uint16)
                all_tokens = np.concatenate((all_tokens, tokenized))

        # test train split
        train = math.ceil(all_tokens.size * .9) # 90% for training
        train_tokens = all_tokens[:train]
        val_tokens = all_tokens[train:]
        splits = ["train", "test"]

        # loading to memmap
        for split in splits:
            tokens = train_tokens if split == "train" else val_tokens
            file = np.memmap(f"{dataset_path}/{split}.bin", dtype=np.uint16, mode="w+", shape=(tokens.size, ))
            num_batches = 16
            shards = np.array_split(tokens, num_batches)
            starting_point = 0

            for batch_id in range(num_batches):
                shard = shards[batch_id]
                file[starting_point:starting_point + shard.size] = shard
            file.flush()
                

# load training set tokens from the memory mapped file
train_dataset = np.memmap(f"{dataset_path}/train.bin", dtype=np.uint16, mode='r')

# same thing for val set
val_dataset = np.memmap(f"{dataset_path}/test.bin", dtype=np.uint16, mode='r')

# each batch is (batch_size, block_size)
total_batch_tokens = (config.batch_size * config.block_size) + 1 # plus one for targets

train_loader = DataLoader(
    train_dataset.astype(np.int32),
    batch_size=total_batch_tokens,
    shuffle=True,
    sampler=None,
)

val_loader = DataLoader(
    val_dataset.astype(np.int32),
    batch_size=total_batch_tokens,
    shuffle=True,
    sampler=None,
)