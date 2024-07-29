import ollama
import json
import re
import argparse

MAX_TEXT_LENGTH = 1600

def generate_qa_from_text(text):
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    prompt = f'请为下面的文本设计5个问答，每个问答之间用换行符隔开，文本如下：\n{text}\n\
你需要遵守下面的格式，注意下面的内容只用来掌握格式，示例如下：\n\
{{"question": "红柯的生长环境和分布地区有哪些？", "answer": "红柯是中国的特有植物，分布在中国大陆的海南等地，生长于海拔350米至1,000米的地区，常生于常绿阔叶林中或海拔较高的山地。"}}\n\
{{"question": "哈卡斯人主要分布在哪里？", "answer": "哈卡斯人主要分布在俄罗斯哈卡斯共和国、部分分布在克拉斯诺亚尔斯克等地，另外还有一部份分布在中国黑龙江省齐齐哈尔市富裕县。"}}\n\
{{"question": "桂城站的出入口及周边有哪些设施？", "answer": "桂城站目前设置的2个出入口均位于南桂东路南侧。本站设有便利店、面包糕饼店、中国银行自动柜员机、自动售货机及“好易”机。"}}\n\
{{"question": "圣地亚哥-罗里盖兹省的名字来源是什么？", "answer": "圣地亚哥-罗里盖兹省的名字来源于建立圣伊纳西奥城镇的军人圣地亚哥-罗里盖兹，他是海地和多明尼加战争中的人物，也是复国战争早期的领导人士之一。"}}\n\
{{"question": "模糊集的隶属函数是什么？", "answer": "模糊集的隶属函数是一个从论域到单位区间的映射，用来表示元素对该集的归属程度。"}}\n\
'
    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']

def validate_qa_format_and_count(qa_pairs):
    qa_pattern = re.compile(r'^\{"question": ".*?", "answer": ".*?"\}$')
    qa_lines = [line.strip() for line in qa_pairs.split('\n') if line.strip()]
    if len(qa_lines) != 5:
        return False
    for line in qa_lines:
        if not qa_pattern.match(line):
            return False
    return True

def read_jsonl(file_path, start_line, end_line):
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            if start_line <= i <= end_line:
                yield json.loads(line)

def save_jsonl(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(input_file, output_file, start_line, end_line):
    for item in read_jsonl(input_file, start_line, end_line):
        text = item.get('text')
        if text:
            while True:
                try:
                    qa_pairs = generate_qa_from_text(text)
                    if validate_qa_format_and_count(qa_pairs):
                        qa_data = [json.loads(line) for line in qa_pairs.split('\n') if line.strip()]
                        save_jsonl(qa_data, output_file)
                        break
                except json.JSONDecodeError:
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from a jsonl file.')
    parser.add_argument('input_file', type=str, help='Path to the input jsonl file')
    parser.add_argument('output_file', type=str, help='Path to the output jsonl file')
    parser.add_argument('start_line', type=int, help='Start line number (inclusive)')
    parser.add_argument('end_line', type=int, help='End line number (inclusive)')
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.start_line, args.end_line)