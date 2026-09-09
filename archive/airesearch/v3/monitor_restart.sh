#!/bin/bash
# 监控 v3_zero_train.log，等 GRPO epoch 1 完成后自动 kill + 重启
LOG="/root/airesearch/v3/v3_zero_train.log"
CKPT_DIR="/root/airesearch/v3/checkpoints_v3_zero"
SCRIPT_DIR="/root/airesearch/v3"
VENV="source /root/airesearch/.torchvenv/bin/activate"
PID_FILE="/tmp/v3_zero_pid"

echo "[monitor] 开始监控 epoch 1 完成..."

while true; do
    # 检查日志中是否包含 epoch 1 完成的标志行
    if grep -q "RL Epoch 1/30" "$LOG" 2>/dev/null; then
        echo "[monitor] ✅ 检测到 Epoch 1 完成！"
        break
    fi
    sleep 10
done

# 再等几秒确保 checkpoint 写完
sleep 5

echo "[monitor] 保存 SFT checkpoint..."
cd "$SCRIPT_DIR"
eval "$VENV"
python3 -c "
import torch
import os

latest = f\"$CKPT_DIR/transformer_cot_v3_zero_latest.pth\"
if os.path.exists(latest):
    ckpt = torch.load(latest, map_location=\"cpu\", weights_only=False)
    # model_state_dict -> sft_best.pth
    torch.save(ckpt[\"model_state_dict\"], f\"$CKPT_DIR/sft_best.pth\")
    print(f\"[monitor] sft_best.pth 已保存 ({os.path.getsize(f\"$CKPT_DIR/sft_best.pth\")} bytes)\")
else:
    print(f\"[monitor] ⚠️ 未找到 {latest}\")
"

# 杀掉当前进程
PID=$(pgrep -f "python3 v3_zero.py" 2>/dev/null | head -1)
if [ -n "$PID" ]; then
    echo "[monitor] 杀掉进程 PID=$PID"
    kill $PID
    sleep 2
    # 确保杀掉
    kill -0 $PID 2>/dev/null && kill -9 $PID
fi

# 重启
echo "[monitor] 重启训练..."
cd "$SCRIPT_DIR"
PYTHONUNBUFFERED=1 nohup python3 v3_zero.py > v3_zero_train.log 2>&1 &
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo "[monitor] ✅ 重启完成，新 PID=$NEW_PID"
