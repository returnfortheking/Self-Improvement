#!/bin/bash
# OpenCode Server启动脚本

echo "==================================="
echo "启动OpenCode HTTP Server"
echo "==================================="
echo ""
echo "服务器配置："
echo "  端口：4096"
echo "  主机名：0.0.0.0（允许局域网访问）"
echo "  CORS：http://localhost:5173（React Native开发）"
echo ""
echo "使用方法："
echo "1. 保持此终端窗口打开"
echo "2. 在React Native应用中配置BASE_URL为http://<你的IP>:4096"
echo "3. 如果使用手机和PC在同一WiFi，使用PC的IP地址"
echo ""
echo "停止：按Ctrl+C"
echo "==================================="
echo ""

# 检查opencode是否安装
if ! command -v opencode &> /dev/null; then
    echo "错误：opencode未安装"
    echo "请先安装opencode："
    echo "  curl -fsSL https://opencode.ai/install | bash"
    echo ""
    exit 1
fi

# 启动OpenCode Server
opencode serve --port 4096 --hostname 0.0.0.0 --cors http://localhost:5173
