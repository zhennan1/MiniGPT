import json

def process_question(question):
    """删除所有问号并在末尾添加或保留一个问号"""
    question = question.replace('？', '') + '？'
    return question

def process_answer(answer):
    """将所有问号替换为句号"""
    answer = answer.replace('？', '。')
    return answer

def process_data(input_file, output_file, log_file):
    """读取输入文件，处理数据并保存到输出文件，记录不符合条件的条目"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    invalid_entries = []

    for entry in data:
        if 'question' in entry:
            original_question = entry['question']
            entry['question'] = process_question(original_question)
            # 检查处理后的 question 是否有且仅有一个问号，并且该问号位于末尾
            if entry['question'].count('？') != 1 or not entry['question'].endswith('？'):
                invalid_entries.append(entry)
        if 'answer' in entry:
            entry['answer'] = process_answer(entry['answer'])

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        for entry in invalid_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

# 输入文件路径、输出文件路径和日志文件路径
input_file = 'C.jsonl'
output_file = 'sft_data_C.jsonl'
log_file = 'invalid_entries_log.jsonl'

# 处理数据
process_data(input_file, output_file, log_file)

print(f"数据处理完成，结果保存在 {output_file}，不符合条件的条目记录在 {log_file}")
