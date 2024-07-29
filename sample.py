import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, MiniGPT
import evaluations
import visualize
import re

# -----------------------------------------------------------------------------

out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.01 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda' # examples: 'cpu', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
eval_mode = True
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------

save_path = os.path.join(out_dir, 'samples')
label_path = os.path.join('.', 'labels.txt')

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
config = GPTConfig(**checkpoint['model_args'])
model = MiniGPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

save_file = open(save_path, 'w')
if eval_mode:
    label_file = open(label_path, 'r')

def remove_repeats(text):
    for length in range(50, 1, -1):
        pattern = re.compile(r'(.{' + str(length) + r'})\1+.*')
        text = pattern.sub(r'\1', text)
    return text

def process_output(output):
    if '<' in output:
        output = output[:output.index('<')]
    if 'A: ' in output:
        output = output[output.index('A: ') + 3:]
    output = output.replace('\uFFFD', '')
    output = output.replace('\n', '')
    output = remove_repeats(output)
    return output

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        starts = [line.strip() for line in f.readlines()]

    if eval_mode:
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]

    rouge_ls = []
    perplexities = []

    for index in range(len(starts)):
        start = starts[index]
        start_ids = encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # run generation
        with torch.no_grad():
            with ctx:
                if not eval_mode:
                    for k in range(num_samples):
                        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        print("Prompt:", start)
                        output_tokens = y[0].tolist()
                        try:
                            end_idx = output_tokens.index(50256)
                            output_tokens = output_tokens[:end_idx]
                        except:
                            pass
                        output = decode(output_tokens)
                        output = process_output(output)
                        print(output)
                        save_file.write(output)
                        print('---------------')
                else:
                    rouge = 0
                    perplexity = 0
                    for k in range(num_samples):
                        y, probs = model.generate_with_probs(x, max_new_tokens, temperature=temperature, top_k=top_k)
                        print("Prompt:", start)
                        output_tokens = y[0].tolist()
                        try:
                            end_idx = output_tokens.index(50256)
                            output_tokens = output_tokens[:end_idx]
                        except:
                            pass
                        output = decode(output_tokens)
                        output = process_output(output)
                        print(output)
                        save_file.write(output + '\n')
                        print('---------------')
                        label = labels[index]
                        rouge += evaluations.rouge_l(output, label)
                        perplexity += evaluations.perplexity(probs)
                    rouge_ls.append(rouge / num_samples)
                    perplexities.append(perplexity / num_samples)
    print('ROUGE-L:', rouge_ls)
    print('Perplexity:', perplexities)
    visualize.visualize_rouge_l(rouge_ls, out_dir)
    visualize.visualize_perplexity(perplexities, out_dir)
else:
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print("Prompt:", start)
                output_tokens = y[0].tolist()
                try:
                    end_idx = output_tokens.index(50256)
                    output_tokens = output_tokens[:end_idx]
                except:
                    pass
                output = decode(output_tokens)
                output = process_output(output)
                print(output)
                save_file.write(output)
                print('---------------')
save_file.close()