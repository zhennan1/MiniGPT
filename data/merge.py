import os
import json

def merge_jsonl_files(input_directory, output_file):
    # 获取目录中的所有文件
    files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files:
            filepath = os.path.join(input_directory, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 写入合并后的文件
                    outfile.write(line)

# Example usage
input_directory = 'sft_data'
output_file = 'sft_data.jsonl'
merge_jsonl_files(input_directory, output_file)
