#!/bin/bash

# 自动等待build_musique完成，然后执行迁移脚本

cd /home/vincent/hyper-simulation

echo "等待 build_musique 完成..."
while pgrep -f "build_musique" > /dev/null; do
    count=$(find data/debug/musique/sample1000 -name "query_hypergraph.pkl" 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] build_musique 仍在运行... 已生成 $count 个 instance"
    sleep 30
done

echo "
============================================================
🎉 build_musique 已完成！
============================================================
"

# 等待一秒确保所有文件都已写入
sleep 1

# 验证生成了文件
count=$(find data/debug/musique/sample1000 -name "query_hypergraph.pkl" 2>/dev/null | wc -l)
echo "✅ 最终生成的 instance 数：$count"

if [ $count -eq 0 ]; then
    echo "❌ 错误：没有生成任何 instance"
    exit 1
fi

echo "
============================================================
开始执行迁移脚本...
============================================================
"

# 执行迁移脚本
pixi run python tests/tasks/migrate_vertex_types.py \
    --source-root data/debug/musique/brk \
    --target-root data/debug/musique/sample1000

echo "
============================================================
✅ 迁移完成！
============================================================
"
