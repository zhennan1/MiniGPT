#!/bin/bash

# 输入文件名
INPUT_FILE="wiki-zh-subset-train_subset.jsonl"
# 每个文件的行数
LINES_PER_FILE=100
# 起始行
START_LINE=12101

# 获取总行数
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
# 计算实际需要处理的行数
LINES_TO_PROCESS=$((TOTAL_LINES - START_LINE + 1))

# 计算需要处理的文件数
NUM_FILES=$((LINES_TO_PROCESS / LINES_PER_FILE))
if [ $((LINES_TO_PROCESS % LINES_PER_FILE)) -ne 0 ]; then
    NUM_FILES=$((NUM_FILES + 1))
fi

# 循环处理每一块
for i in $(seq 0 $((NUM_FILES - 1))); do
    CURRENT_START_LINE=$((START_LINE + i * LINES_PER_FILE))
    END_LINE=$((CURRENT_START_LINE + LINES_PER_FILE - 1))
    
    # 确保END_LINE不超过总行数
    if [ $END_LINE -gt $TOTAL_LINES ]; then
        END_LINE=$TOTAL_LINES
    fi
    
    # 生成输出文件名
    OUTPUT_FILE="output_${CURRENT_START_LINE}_${END_LINE}.jsonl"
    
    # 调用Python脚本处理当前块
    python generate.py "$INPUT_FILE" "$OUTPUT_FILE" "$CURRENT_START_LINE" "$END_LINE"
done
