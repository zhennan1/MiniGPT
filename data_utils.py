import os
import torch
import tiktoken
import numpy as np

# Initialize the tiktoken encoder
enc = tiktoken.get_encoding("gpt2")

train_data = None
val_data = None

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask

def get_batch_sft(split, batch_size, block_size, device): 
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    
    # Select random batch
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    
    # Initialize x, y and loss_mask tensors
    x = torch.full((batch_size, block_size), enc.encode_ordinary('<pad>')[0], dtype=torch.int64)
    y = torch.full((batch_size, block_size), enc.encode_ordinary('<pad>')[0], dtype=torch.int64)
    loss_mask = torch.zeros((batch_size, block_size), dtype=torch.float64)
    
    for i, idx in enumerate(ix):
        sequence = data[idx:idx+block_size+1].astype(np.int64)
        
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        x[i, :len(input_seq)] = torch.tensor(input_seq, dtype=torch.int64)
        y[i, :len(target_seq)] = torch.tensor(target_seq, dtype=torch.int64)
        loss_mask[i, :len(target_seq)] = 1.0  # Set loss mask only for the length of the target sequence
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    
    return x, y, loss_mask
