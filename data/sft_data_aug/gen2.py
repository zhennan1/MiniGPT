import json
import ollama

# Function to generate 10 variations of a given question
def generate_question_variants(question, answer):
    prompt = f'根据下面的答案：\n{answer}\n\n改写下面的问题，但不要改变原意，每个编辑距离<5，不能和原来的内容一样，输出10行，每行为1个问题。原问题如下：\n{question}'
    response = ollama.generate(model='llama3.1', prompt=prompt)
    # Split the response into lines
    variants = response['response'].split('\n')
    # Ensure we have exactly 10 lines
    if len(variants) == 10:
        return variants
    else:
        raise ValueError("Response does not contain exactly 10 lines")

# Function to read questions from A.jsonl, generate variants, and save to B.jsonl
def process_questions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'a', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            question = data['question']
            answer = data['answer']
            while True:
                try:
                    variants = generate_question_variants(question, answer)
                    for variant in variants:
                        new_data = {
                            "question": variant,
                            "answer": answer
                        }
                        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                    break  # Exit the loop if successful
                except ValueError as e:
                    print(f"Error processing question: {question} - {str(e)}, retrying...")

# Main execution
input_file = 'A.jsonl'
output_file = 'C.jsonl'
process_questions(input_file, output_file)
