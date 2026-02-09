#!/bin/bash

###############################################################################
# OpenCode Server 停止脚本 (macOS)
###############################################################################

PID_FILE="$HOME/opencode-server.pid"

echo "停止 OpenCode Server..."

if [ ! -f "$PID_FILE" ]; then
    echo "未找到 PID 文件，OpenCode Server 可能没有在运行"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "发送 TERM 信号到进程 $PID..."
    kill "$PID"

    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "✓ OpenCode Server 已停止"
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
    done

    # 如果还没停止，强制杀死
    echo "强制停止进程 $PID..."
    kill -9 "$PID"
    rm -f "$PID_FILE"
    echo "✓ OpenCode Server 已强制停止"
else
    echo "进程 $PID 不存在，清理 PID 文件..."
    rm -f "$PID_FILE"
fi
