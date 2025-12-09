#!/bin/bash

# 项目路径
PROJECT_DIR="/Users/fuzhenhao/Desktop/Vibe Coding/quant_playground/projects/investment-assistant"

# 虚拟环境路径
VENV_PATH="/Users/fuzhenhao/Desktop/Vibe Coding/quant_playground/projects/.venv/bin/activate"

echo ">>> 激活虚拟环境..."
source "$VENV_PATH"

echo ">>> 切换目录到项目路径..."
cd "$PROJECT_DIR" || exit

echo ">>> 添加所有更改..."
git add .

# 生成默认 commit 信息（带时间戳）
COMMIT_MSG="auto commit $(date '+%Y-%m-%d %H:%M:%S')"
echo ">>> 提交代码：$COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo ">>> 推送到 GitHub..."
git push origin main

echo ">>> 完成！"
