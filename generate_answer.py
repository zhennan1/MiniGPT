import json
from gradio_client import Client

client1 = Client("http://127.0.0.1:7860/")

def gen_client(model_choice, prompt, temperature):
    result1 = client1.predict(
        model_choice=model_choice,
        prompt=prompt,
        temperature=temperature,
        api_name="/predict"
    )
    return result1

# 读取输入文件
input_filename = '测试集-day1.jsonl'
output_filename = '2021030014.jsonl'

with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        question = data["question"]
        
        # 调用API生成答案
        answer = gen_client("SFT", question, 0.01)
        
        # 构建输出JSONL格式
        output_data = {
            "question": question,
            "answer": answer
        }
        
        # 写入输出文件
        outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

print(f"输出已保存到 {output_filename}")
