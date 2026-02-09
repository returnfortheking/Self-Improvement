#!/bin/bash

###############################################################################
# OpenCode Server 启动脚本 (macOS)
#
# 功能：
# - 启动 OpenCode Server
# - 自动重启（如果崩溃）
# - 日志记录
###############################################################################

# 配置
OPENCODE_PORT=4096
OPENCODE_HOST=0.0.0.0
LOG_DIR="$HOME/Library/Logs/Opencode"
PID_FILE="$HOME/opencode-server.pid"
LOG_FILE="$LOG_DIR/opencode-server.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "OpenCode Server 启动脚本"
echo "=========================================="
echo ""

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "OpenCode Server 已经在运行 (PID: $OLD_PID)"
        echo "如需重启，请先运行: ./stop-opencode.sh"
        exit 1
    else
        echo "清理旧的 PID 文件..."
        rm -f "$PID_FILE"
    fi
fi

# 启动 OpenCode Server
echo "启动 OpenCode Server..."
echo "端口: $OPENCODE_PORT"
echo "主机: $OPENCODE_HOST"
echo "日志: $LOG_FILE"
echo ""

# 使用 nohup 在后台运行
nohup opencode serve \
    --port "$OPENCODE_PORT" \
    --hostname "$OPENCODE_HOST" \
    >> "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"

# 等待几秒检查是否启动成功
sleep 3

if ps -p $PID > /dev/null; then
    echo "✓ OpenCode Server 启动成功!"
    echo "  PID: $PID"
    echo "  日志: tail -f $LOG_FILE"
    echo ""
    echo "测试连接:"
    echo "  curl http://localhost:$OPENCODE_PORT/global/health"
else
    echo "✗ OpenCode Server 启动失败!"
    echo "请查看日志: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
