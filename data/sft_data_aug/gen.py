import ollama
import json
import re
import argparse
import os

MAX_TEXT_LENGTH = 1600
OUTPUT_FILE = 'qa_output.txt'

def generate_qa_from_text():
    prompt = f'改写下面的问题和答案，每个编辑距离<5，不能和原来的内容一样，不要改变原意，不要改变格式\n\
{{"question": "什么是金星凌日？", "answer": "金星凌日是当金星运行到太阳和火星之间时发生的一种罕见的天文现象。"}}\n\
{{"question": "从火星上看金星凌日会是什么样子？", "answer": "从火星上可以看到金星像一个黑色圆盘从太阳表面划过。"}}\n\
{{"question": "计算金火凌日周期的公式是什么？", "answer": "金火凌日的周期是运用公式1/(1/P-1/Q)运算的。"}}\n\
{{"question": "短鲬属于哪一类鱼？", "answer": "短鲬属于辐鳍鱼纲鲉形目短鲬科短鲬属的鱼类。"}}\n\
{{"question": "短鲬分布于哪些海域？", "answer": "短鲬分布于西北太平洋区，包括朝鲜、日本以及中国东海、黄海、渤海等海域。"}}\n\
{{"question": "什么是时钟频率？", "answer": "时钟频率是指同步电路中时钟的基础频率，它以“每秒时钟周期”来度量，量度单位采用SI单位赫兹（Hz）。"}}\n\
{{"question": "时钟频率的量度单位是什么？", "answer": "时钟频率的量度单位是赫兹（Hz）。"}}\n\
{{"question": "俄土战争（1676年-1681年）持续了多久？", "answer": "战争持续了5年。"}}\n\
{{"question": "俄土战争（1676年-1681年）是如何结束的？", "answer": "双方于1681年签署巴赫奇萨赖和约，奥斯曼帝国承认沙皇俄国对第聂伯河左岸地区的统治。"}}\n\
{{"question": "在力学里，自由度指的是什么？", "answer": "自由度指的是力学系统的独立坐标的个数。"}}\n\
'
    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']

def validate_qa_format_and_count(qa_pairs):
    qa_pattern = re.compile(r'^\{"question": ".*?", "answer": ".*?"\}$')
    qa_lines = [line.strip() for line in qa_pairs.split('\n') if line.strip()]
    if len(qa_lines) != 10:
        return False
    for line in qa_lines:
        if not qa_pattern.match(line):
            return False
    return True

def save_jsonl(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    while True:
        try:
            qa_pairs = generate_qa_from_text()
            if validate_qa_format_and_count(qa_pairs):
                qa_data = [json.loads(line) for line in qa_pairs.split('\n') if line.strip()]
                save_jsonl(qa_data, OUTPUT_FILE) 
                print(f'Results have been appended to {OUTPUT_FILE}')
        except json.JSONDecodeError:
            continue

if __name__ == '__main__':
    main()
