import json

# 文件路径
input_file_path = 'A.jsonl'
questions_file_path = 'questions.txt'
answers_file_path = 'answers.txt'

# 打开文件
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(questions_file_path, 'w', encoding='utf-8') as questions_file, \
     open(answers_file_path, 'w', encoding='utf-8') as answers_file:
    
    for line in input_file:
        # 解析json
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        
        # 写入文件
        questions_file.write(question + '\n')
        answers_file.write(answer + '\n')

print("转换完成")
