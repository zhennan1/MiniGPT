### TODO: prepare SFT data similar to `prepare.py`
###

import os
import sys
import json
import tiktoken
import numpy as np

# Initialize the tiktoken encoder
enc = tiktoken.get_encoding("gpt2")

# Read the input filenames from command-line arguments
names = sys.argv[1:]

### Read data from the specified JSONL files
data = []
for name in names:
    with open(name + ".jsonl") as f:
        for line in f:
            entry = json.loads(line)
            answer = entry['answer']
            answer_tokens = enc.encode_ordinary(answer)
            answer_tokens.append(enc.eot_token)  # Add end-of-text token
            encoded_answer = enc.decode(answer_tokens)
            data.append(f"Q: {entry['question']}\nA: {encoded_answer}\n")

# Combine all entries into a single string
data = '\n'.join(data)

# Split data into training (90%) and validation (10%) sets
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

### Tokenize the raw data using the tiktoken encoder and convert to numpy arrays
train_ids = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)

# Create a directory to save the processed data if it doesn't already exist
os.makedirs("processed_sft", exist_ok=True)

# Save the numpy arrays to binary files
train_ids.tofile(os.path.join("processed_sft", "train.bin"))
val_ids.tofile(os.path.join("processed_sft", 'val.bin'))
