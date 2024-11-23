#python fineweb.py
#WIll save shards to the local directory "edu_fineweb10B"

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

#----------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) #100M tokens per shard, total of 100 shards

#create the cache of the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok = True)

#download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

#init the tokenizer
enc = tiktoken.get_encoding("gpt2")