import pandas as pd
import numpy as np
import tiktoken
from config import ConfigClass as config
import math

# for models with bigger and more complex inputs, use newer tokenizers (such as gpt-4)
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

# modify gpt2 tokenizer to add special tokens
special_tokens = {"<|pad|>": 50258}
if not config.pretraining:
    special_tokens["<|user|>"] = 50259
    special_tokens["<|assistant|>"] = 50260

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
tokenizer = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="gpt2_with_pad",
    pat_str=gpt2_tokenizer._pat_str,
    mergeable_ranks=gpt2_tokenizer._mergeable_ranks,
    special_tokens={
        **gpt2_tokenizer._special_tokens,
        **special_tokens,
    }
)

file_reader = pd.read_csv("./data/finetuning/data.csv", chunksize=1000)

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
splits = ["train", "val"]

# loading to memmap
for split in splits:
    tokens = train_tokens if split == "train" else val_tokens
    file = np.memmap(f"./data/custom/{split}.bin", dtype=np.uint16, mode="w+", shape=(tokens.size, ))
    num_batches = 16
    shards = np.array_split(tokens, num_batches)
    starting_point = 0

    for batch_id in range(num_batches):
        shard = shards[batch_id]
        file[starting_point:starting_point + shard.size] = shard
    file.flush()
