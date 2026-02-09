# Mac mini 24小时部署指南

## 📋 概述

本指南说明如何在 Mac mini 上部署 OpenCode Server，实现 24小时不间断运行，支持移动端应用随时连接。

## 🔧 前置要求

### 1. 系统要求
- macOS 11 (Big Sur) 或更高版本
- 至少 2GB 可用内存
- 稳定的网络连接

### 2. 安装必要工具

```bash
# 安装 Homebrew（如果未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Node.js
brew install node

# 安装 OpenCode CLI（根据实际安装方式）
npm install -g @opencode/cli

# 安装 ngrok
brew install ngrok/ngrok/ngrok

# 配置 ngrok（需要 ngrok 账号）
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### 3. 安装 OpenCode Server

根据你的实际安装方式，确保 `opencode` 命令可用：

```bash
# 测试 OpenCode CLI
opencode --version

# 测试 Server 启动
opencode serve --help
```

## 📦 部署步骤

### 步骤 1: 准备部署脚本

```bash
# 1. 克隆或复制项目到 Mac mini
cd ~/Projects/mobile_skills_mvp

# 2. 设置脚本执行权限
chmod +x mac-mini-deploy/start-opencode.sh
chmod +x mac-mini-deploy/stop-opencode.sh
chmod +x mac-mini-deploy/start-ngrok.sh

# 3. 编辑 launchd plist 文件（替换 YOUR_USERNAME）
# 使用 nano 或 vim 编辑:
# nano mac-mini-deploy/com.opencode.server.plist
```

在 `com.opencode.server.plist` 中，将 `YOUR_USERNAME` 替换为你的实际用户名：

```xml
<string>/Users/YOUR_USERNAME/Projects/mobile_skills_mvp/mac-mini-deploy/start-opencode.sh</string>
```

### 步骤 2: 配置开机自动启动

```bash
# 1. 复制 plist 文件到 LaunchAgents
cp mac-mini-deploy/com.opencode.server.plist ~/Library/LaunchAgents/

# 2. 加载服务
launchctl load ~/Library/LaunchAgents/com.opencode.server.plist

# 3. 启动服务
launchctl start com.opencode.server

# 4. 检查服务状态
launchctl list | grep opencode
```

### 步骤 3: 启动服务

```bash
# 方式 1: 使用 launchd（推荐）
# 服务会自动启动

# 方式 2: 手动启动（测试用）
cd mac-mini-deploy
./start-opencode.sh
./start-ngrok.sh
```

### 步骤 4: 验证部署

```bash
# 1. 检查 OpenCode Server
curl http://localhost:4096/global/health

# 预期输出: {"healthy":true,"version":"..."}

# 2. 检查 ngrok 隧道
cat ~/ngrok-tunnel-url.txt

# 3. 测试公网访问
curl $(cat ~/ngrok-tunnel-url.txt)/global/health

# 4. 查看日志
tail -f ~/Library/Logs/Opencode/opencode-server.log
tail -f ~/Library/Logs/Opencode/ngrok.log
```

### 步骤 5: 配置移动端应用

1. 获取 ngrok 隧道 URL：
```bash
cat ~/ngrok-tunnel-url.txt
# 输出示例: https://abc123.ngrok-free.dev
```

2. 更新移动端服务器配置：

编辑 `src/services/servers.config.ts`:

```typescript
export const DEFAULT_SERVERS: ServerConfig[] = [
  {
    name: 'Mac Mini',
    url: 'https://abc123.ngrok-free.dev', // 替换为实际 URL
    priority: 1,
    enabled: true,
  },
  {
    name: 'Windows PC',
    url: 'https://windows-pc.ngrok-free.dev', // 备用服务器
    priority: 2,
    enabled: true,
  },
];
```

3. 重新构建并安装 APK

## 🔍 监控和维护

### 查看服务状态

```bash
# OpenCode Server 进程
ps aux | grep "opencode serve"

# Ngrok 进程
ps aux | grep ngrok

# 端口监听
lsof -i :4096
```

### 查看日志

```bash
# OpenCode Server 日志
tail -f ~/Library/Logs/Opencode/opencode-server.log

# Ngrok 日志
tail -f ~/Library/Logs/Opencode/ngrok.log

# launchd 日志
tail -f ~/Library/Logs/Opencode/opencode-launchd.log
```

### 重启服务

```bash
# 停止服务
cd mac-mini-deploy
./stop-opencode.sh

# 停止 ngrok
kill $(cat ~/ngrok.pid)

# 启动服务
./start-opencode.sh
./start-ngrok.sh
```

### 更新 OpenCode Server

```bash
# 1. 停止服务
./stop-opencode.sh

# 2. 更新 CLI
npm update -g @opencode/cli

# 3. 重新启动
./start-opencode.sh
```

## ⚠️ 故障排查

### 问题 1: 服务无法启动

**检查**:
```bash
# 查看详细错误
cat ~/Library/Logs/Opencode/opencode-server.log

# 检查端口占用
lsof -i :4096
```

**解决**:
- 杀死占用端口的进程: `kill -9 <PID>`
- 修改端口（如果 4096 被占用）

### 问题 2: ngrok 隧道无法建立

**检查**:
```bash
# 查看 ngrok 日志
tail -f ~/Library/Logs/Opencode/ngrok.log

# 测试 ngrok 配置
ngrok config check
```

**解决**:
- 确保 ngrok 已正确配置 authtoken
- 检查网络连接
- 重启 ngrok: `./start-ngrok.sh`

### 问题 3: 移动端无法连接

**检查**:
```bash
# 1. 在 Mac mini 上测试本地连接
curl http://localhost:4096/global/health

# 2. 测试 ngrok 隧道
curl $(cat ~/ngrok-tunnel-url.txt)/global/health

# 3. 检查防火墙设置
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

**解决**:
- 确保 OpenCode Server 监听 0.0.0.0（不是 127.0.0.1）
- 检查 ngrok 免费版限制（可能需要重启隧道）
- 更新移动端配置中的服务器 URL

### 问题 4: launchd 服务不自动启动

**检查**:
```bash
# 查看服务状态
launchctl list | grep opencode

# 查看错误日志
log show --predicate 'process == "opencode"' --last 1h
```

**解决**:
```bash
# 卸载服务
launchctl unload ~/Library/LaunchAgents/com.opencode.server.plist

# 重新加载
launchctl load ~/Library/LaunchAgents/com.opencode.server.plist

# 启动
launchctl start com.opencode.server
```

## 🔒 安全建议

1. **ngrok 免费版限制**:
   - 随机 URL，每次重启可能变化
   - 考虑升级到付费版获取固定域名

2. **访问控制**:
   - OpenCode Server 默认无认证
   - 考虑使用反向代理（nginx）添加基本认证

3. **日志轮转**:
   - 定期清理日志文件
   - 或使用 logrotate 管理日志大小

4. **网络监控**:
   - 定期检查异常访问
   - 监控带宽使用

## 📊 性能优化

### 资源限制

```bash
# 查看当前资源使用
top -pid $(cat ~/opencode-server.pid)

# 如果内存使用过高，考虑:
# 1. 限制 OpenCode Server 内存
# 2. 定期重启服务
```

### 自动化维护

创建定时任务（cron）清理日志：

```bash
# 编辑 crontab
crontab -e

# 添加每天凌晨 2 点清理日志
0 2 * * * rm ~/Library/Logs/Opencode/*.log
```

## 📞 支持

- OpenCode 文档: [官方文档链接]
- ngrok 文档: https://ngrok.com/docs
- 项目 Issues: [GitHub Issues]

## ✅ 部署检查清单

- [ ] 已安装 Node.js 和 npm
- [ ] 已安装 OpenCode CLI
- [ ] 已安装 ngrok 并配置
- [ ] 脚本已设置执行权限
- [ ] launchd plist 已配置正确的用户名
- [ ] 服务已加载到 launchd
- [ ] OpenCode Server 正常运行（端口 4096）
- [ ] ngrok 隧道已建立
- [ ] 本地测试通过（curl localhost:4096）
- [ ] 公网测试通过（curl ngrok-url）
- [ ] 移动端配置已更新
- [ ] 移动端成功连接

---

**部署完成后，你的 Mac mini 将成为 24小时运行的 OpenCode Server！** 🎉
