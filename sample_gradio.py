from time import sleep
import gradio as gr
import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, MiniGPT
import re

# -----------------------------------------------------------------------------

start = "\n" # or "" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 256 # number of tokens generated in each sample
temperature = 0.01 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1234
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file

# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model(out_dir):
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
    return model

# 初始加载默认模型
out_dir = 'out-pretrain' # 可以根据需要更改默认模型目录
model = load_model(out_dir)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

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

def generate_text(model_choice, prompt, temperature=0.5):
    global model
    global out_dir
    
    # 根据选择的模型加载相应的模型
    if model_choice == "Pretrain":
        out_dir = 'out-pretrain'
    elif model_choice == "SFT":
        out_dir = 'out-sft-mix'
    
    model = load_model(out_dir)

    start_ids = encode(prompt)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            output_tokens = y[0].tolist()
            if 50256 in output_tokens:  # Check for end of text token
                end_idx = output_tokens.index(50256)
                output_tokens = output_tokens[:end_idx]
            output = decode(output_tokens)
            output = process_output(output)
            out = ""
            for ch in output:
                out = out + ch
                yield out
                sleep(0.05)

temperature_slider = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, label="Temperature")
iface = gr.Interface(
    fn=generate_text,
    inputs=[gr.components.Radio(["Pretrain", "SFT"], label="Model Choice"),
            gr.components.Textbox(lines=2, placeholder="Enter your prompt here..."), 
            temperature_slider],
    outputs=gr.components.Textbox(),
    title="MiniGPT Text Generation",
    description="Enter a prompt and select a model to get generated text from the MiniGPT model.",
)

if __name__ == "__main__":
    iface.launch(share=True)
