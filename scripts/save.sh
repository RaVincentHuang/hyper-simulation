#!/bin/bash

# ================= 配置 =================
export LD_LIBRARY_PATH="/home/vincent/hyper-simulation/.pixi/envs/simulation/lib"

# 结果输出目录
RESULT_DIR="data/baseline"
mkdir -p $RESULT_DIR

# 数据集列表 (pixi 任务名 → 小写任务名映射)
declare -A TASK_MAP=(
    ["HotpotQA"]="hotpotqa"
    ["Musique"]="musique"
    ["MultiHop"]="multihop"
    ["ARC"]="arc"
    ["LegalBench"]="legalbench"
)

# 方法列表（包含所有需要跑的基线方法）
METHODS=("vanilla" "contradoc" "cdit" "sparsecl" "sentli")

# ================= 主循环 =================
echo "🚀 开始生成结果"
echo "📂 结果输出目录：$RESULT_DIR"

for method in "${METHODS[@]}"; do
    echo ""
    echo "=========================================="
    echo "🔹 方法：$method"
    echo "=========================================="
    
    mkdir -p $RESULT_DIR/$method
    
    for dataset in "${!TASK_MAP[@]}"; do
        task_lower="${TASK_MAP[$dataset]}"
        
        echo ""
        echo "📦 处理：$dataset"
        # 直接运行的方法
        echo "   ⚡ 模式: 直接运行预处理并生成"
        
        pixi run -e simulation $dataset \
            --output_path $RESULT_DIR/$method \
            --method $method
        
        # 检查结果（根据 rag_no_retrival.py，输出文件名就是 task_lower.json）
        output_file="$RESULT_DIR/$method/${task_lower}.json"
        if [ -f "$output_file" ]; then
            echo "   ✅ 完成：$output_file"
        else
            echo "   ⚠️  结果文件未找到，请检查日志"
        fi
    done
done

echo ""
echo "=========================================="
echo "🎉 所有生成任务完成！"
echo "📂 结果目录：$RESULT_DIR"
echo "=========================================="