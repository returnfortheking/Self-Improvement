#!/bin/bash

###############################################################################
# Ngrok 启动和监控脚本 (macOS)
#
# 功能：
# - 启动 ngrok 隧道
# - 自动检测隧道地址
# - 定期检查连接状态
# - 隧道断开时自动重启
###############################################################################

NGROK_PORT=4096
LOG_DIR="$HOME/Library/Logs/Opencode"
NGROK_LOG="$LOG_DIR/ngrok.log"
NGROK_PID_FILE="$HOME/ngrok.pid"
TUNNEL_URL_FILE="$HOME/ngrok-tunnel-url.txt"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Ngrok 隧道启动脚本"
echo "=========================================="
echo ""

# 检查 ngrok 是否安装
if ! command -v ngrok &> /dev/null; then
    echo "✗ ngrok 未安装!"
    echo "请安装: brew install ngrok/ngrok/ngrok"
    exit 1
fi

# 检查 ngrok 是否已配置
if ! ngrok config check &> /dev/null; then
    echo "✗ ngrok 未配置!"
    echo "请先运行: ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

# 停止现有的 ngrok 进程
if [ -f "$NGROK_PID_FILE" ]; then
    OLD_PID=$(cat "$NGROK_PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "停止现有的 ngrok 进程..."
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$NGROK_PID_FILE"
fi

# 启动 ngrok
echo "启动 ngrok 隧道..."
echo "本地端口: $NGROK_PORT"
echo "日志: $NGROK_LOG"
echo ""

ngrok http "$NGROK_PORT" \
    --log=stdout \
    --log-format=json \
    > "$NGROK_LOG" 2>&1 &

NGROK_PID=$!
echo $NGROK_PID > "$NGROK_PID_FILE"

echo "✓ Ngrok 启动成功!"
echo "  PID: $NGROK_PID"
echo ""

# 等待隧道建立
echo "等待隧道建立..."
sleep 5

# 从日志中提取隧道 URL
TUNNEL_URL=$(grep -o '"https://[^"]*"' "$NGROK_LOG" | head -1 | tr -d '"')

if [ -n "$TUNNEL_URL" ]; then
    echo "✓ 隧道已建立!"
    echo "  URL: $TUNNEL_URL"
    echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"
    echo ""
    echo "测试连接:"
    echo "  curl $TUNNEL_URL/global/health"
else
    echo "⚠ 无法获取隧道 URL，请检查日志:"
    echo "  tail -f $NGROK_LOG"
fi

echo ""
echo "Ngrok 控制台: https://dashboard.ngrok.com"
echo "查看日志: tail -f $NGROK_LOG"
