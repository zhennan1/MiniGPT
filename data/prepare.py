import os
import sys
import json
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

### TODO: read data from ([name].jsonl for name in names)
### TODO: combine multiple files(if needed) into one single data file
### TODO: split data for train(0.9) and valid (0.1)

data = []
for name in names:
    with open(name + ".jsonl") as f:
        for line in f:
            data.append(json.loads(line)["text"])

data = '\n'.join(data)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)
###

# save numpy array to file [name]/train.bin and [name]/val.bin
os.makedirs("processed_pretrain", exist_ok=True)
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
