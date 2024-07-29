import ollama
import json
import re
import argparse
import os

MAX_TEXT_LENGTH = 1600
OUTPUT_FILE = 'qa_output1.txt'

def generate_qa_from_text():
    prompt = f'改写下面的问题和答案，每个编辑距离<5，不能和原来的内容一样，不要改变原意，不要改变格式\n\
{{"question": "红柯的生长环境和分布地区有哪些？", "answer": "红柯是中国的特有植物，分布在中国大陆的海南等地，生长于海拔350米至1,000米的地区，常生于常绿阔叶林中或海拔较高的山地。"}}\n\
{{"question": "韩林儿建立的宋政权是什么时候开始和结束的？", "answer": "韩林儿建立的宋政权始于1355年，结束于1366年。"}}\n\
{{"question": "三波春夫在浪曲和歌谣界的代表作和成就有哪些？", "answer": "三波春夫以浪曲师的身份有取材于浪曲题材的歌谣浪曲最为得意，特别是以元禄名枪谱俵星玄番为代表的长篇歌谣浪曲方面，三波的演绎最为受到关注和好评。"}}\n\
{{"question": "黄缘萤的栖息地和成虫习性有哪些？", "answer": "黄缘萤栖息于静止的水域，其幼虫亦是水生。成虫整年可见，但在台湾于4月及8月最常见。成虫夜行，平时并不觅食，但会以露水解渴。"}}\n\
{{"question": "哈卡斯人主要分布在哪里？", "answer": "哈卡斯人主要分布在俄罗斯哈卡斯共和国、部分分布在克拉斯诺亚尔斯克等地，另外还有一部份分布在中国黑龙江省齐齐哈尔市富裕县。"}}\n\
{{"question": "东南亚福建话有哪些特点？", "answer": "东南亚福建话保留了许多古雅的词汇和用法，同时由于早期带过去的语言，向东南亚本地语言和方言借词，形成多种不同掺杂了土语词汇的闽南语。"}}\n\
{{"question": "鹅膏菌属中有哪些著名的可食用菇类？", "answer": "鹅膏菌属中一些著名的可食用菇类包括中部非洲的Amanita zambiana、墨西哥的A. basii、欧洲的白橙盖鹅膏和东南亚的白条盖鹅膏菌。"}}\n\
{{"question": "德意志国铁路01型蒸汽机车的轮式和最高时速分别是多少？", "answer": "德意志国铁路01型蒸汽机车的轮式为4-6-2型，最高时速最初为120千米，后增加到130千米。"}}\n\
{{"question": "桂城站的出入口及周边有哪些设施？", "answer": "桂城站目前设置的2个出入口均位于南桂东路南侧。本站设有便利店、面包糕饼店、中国银行自动柜员机、自动售货机及“好易”机。"}}\n\
{{"question": "泥柯的分布地区和生长环境是什么？", "answer": "泥柯分布在印度、缅甸以及中国大陆的西藏、云南等地，生长于海拔1,700米的地区，常生于山地常绿阔叶林中，目前尚未由人工引种栽培。"}}\n\
{{"question": "2012年欧洲冠军联赛决赛的最终比分是多少？", "answer": "切尔西在点球大战中以4:3赢得比赛，历史上首次捧起欧洲冠军联赛奖杯。"}}\n\
{{"question": "圣彼德省的主要经济产业是什么？", "answer": "圣彼德省的制糖业兴盛。"}}\n\
{{"question": "圣地亚哥-罗里盖兹省的名字来源是什么？", "answer": "圣地亚哥-罗里盖兹省的名字来源于建立圣伊纳西奥城镇的军人圣地亚哥-罗里盖兹，他是海地和多明尼加战争中的人物，也是复国战争早期的领导人士之一。"}}\n\
{{"question": "巴韦德省何时脱离圣地牙哥省成为独立省份？", "answer": "巴韦德省在1958年脱离圣地牙哥省。"}}\n\
{{"question": "萨曼莎·斯托瑟在2011年获得了哪个大满贯赛事的女单冠军？", "answer": "萨曼莎·斯托瑟在2011年美国网球公开赛女单决赛中击败塞雷娜·威廉姆斯，获得个人第一座大满贯单打冠军。"}}\n\
{{"question": "普雷斯堡路位于巴黎的哪些区？", "answer": "普雷斯堡路位于巴黎的第八区和第十六区。"}}\n\
{{"question": "蒂尔西特路是以什么命名的？", "answer": "蒂尔西特路以蒂尔西特和约命名（蒂尔西特是苏维埃茨克原来的德语名称）。"}}\n\
{{"question": "黑带天竺鲷的模式产地在哪里？", "answer": "黑带天竺鲷的模式产地在Bikini、Atoll、马绍尔群岛。"}}\n\
{{"question": "拟双线天竺鲷有哪些俗名？", "answer": "拟双线天竺鲷的俗名是拟双带天竺鲷。"}}\n\
{{"question": "模糊集的隶属函数是什么？", "answer": "模糊集的隶属函数是一个从论域到单位区间的映射，用来表示元素对该集的归属程度。"}}\n\
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
