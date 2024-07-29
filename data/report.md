# MiniGPT

万振南 计23 2021030014

## 一、数据集和预处理

### 1. 预训练数据集

中文wiki百科数据集的一个子集：wiki-zh-subset-train_subset.jsonl

下载链接：https://cloud.tsinghua.edu.cn/d/d6f9966d6b684b05a516/

### 2. 微调数据集

**sft_data**



### 3. 数据预处理

文本编码器：tiktoken.get_encoding("gpt2")

文本编码：enc.encode_ordinary(data)

训练集和验证集划分：90%用作训练集，10%用作验证集

数据保存：使用 NumPy 将编码后的文本数据保存为二进制文件，训练集和验证集分别保存在名为 train.bin 和 val.bin 的文件中