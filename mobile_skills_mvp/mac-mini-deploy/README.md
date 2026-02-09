# Mac mini 24小时部署方案

## 🎯 目标

将 OpenCode Server 部署在 Mac mini 上，实现：
- ✅ 24小时不间断运行
- ✅ 移动端随时可访问
- ✅ 自动故障转移（Mac mini → Windows PC）
- ✅ 开机自动启动
- ✅ 崩溃自动恢复

## 🏗️ 架构设计

```
┌─────────────────┐
│  移动端应用      │
│ (Android)       │
└────────┬────────┘
         │
         ├──────┐
         │      │
         ▼      ▼
┌─────────────┐  ┌──────────────┐
│  Mac mini   │  │  Windows PC  │
│  (主服务器)  │  │  (备用服务器) │
│  优先级: 1   │  │  优先级: 2    │
└─────────────┘  └──────────────┘
     │                    │
     ▼                    ▼
OpenCode Server     OpenCode Server
  + ngrok              + ngrok
  Port: 4096          Port: 4096
  24小时运行           按需运行
```

## 📁 文件结构

```
mac-mini-deploy/
├── README.md                  # 本文件
├── DEPLOYMENT_GUIDE.md        # 详细部署指南
├── start-opencode.sh          # OpenCode Server 启动脚本
├── stop-opencode.sh           # OpenCode Server 停止脚本
├── start-ngrok.sh             # Ngrok 启动脚本
└── com.opencode.server.plist  # launchd 配置文件
```

## 🚀 快速开始

### 1. 一键部署（Mac mini）

```bash
# 1. 设置执行权限
chmod +x mac-mini-deploy/*.sh

# 2. 编辑 launchd 配置（替换用户名）
nano mac-mini-deploy/com.opencode.server.plist

# 3. 安装服务
cp mac-mini-deploy/com.opencode.server.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.opencode.server.plist

# 4. 启动服务
launchctl start com.opencode.server
```

### 2. 配置移动端

编辑 `src/services/servers.config.ts`，添加 Mac mini ngrok 地址：

```typescript
export const DEFAULT_SERVERS: ServerConfig[] = [
  {
    name: 'Mac Mini',
    url: 'https://your-mac-mini.ngrok-free.dev',
    priority: 1,
    enabled: true,
  },
  {
    name: 'Windows PC',
    url: 'https://your-windows-pc.ngrok-free.dev',
    priority: 2,
    enabled: true,
  },
];
```

## 🔍 工作原理

### 1. 服务器管理（ServerManager）

移动端应用使用 `ServerManager` 类管理多个服务器：

```typescript
// 自动选择最佳服务器
const server = await serverManager.selectBestServer();

// 健康检查
await serverManager.checkAllServers();

// 故障转移
await serverManager.handleServerFailure(failedServer);
```

**优先级逻辑**：
1. Mac Mini (优先级 1) - 首选
2. Windows PC (优先级 2) - 备用

### 2. 开机自启动（launchd）

使用 macOS launchd 实现：

```xml
<key>RunAtLoad</key>
<true/>

<key>KeepAlive</key>
<dict>
    <key>Crashed</key>
    <true/>
</dict>
```

**特性**：
- 系统启动时自动运行
- 崩溃后自动重启
- 日志记录到标准位置

### 3. Ngrok 隧道

```bash
# 启动 ngrok 隧道
ngrok http 4096 --log=stdout --log-format=json

# 自动获取隧道 URL
TUNNEL_URL=$(grep -o '"https://[^"]*"' ngrok.log | head -1)
```

**特性**：
- 公网访问
- 自动断线重连
- 日志记录

## 📊 监控和维护

### 健康检查

```bash
# OpenCode Server
curl http://localhost:4096/global/health

# Ngrok 隧道
curl https://your-mac-mini.ngrok-free.dev/global/health
```

### 日志查看

```bash
# OpenCode Server 日志
tail -f ~/Library/Logs/Opencode/opencode-server.log

# Ngrok 日志
tail -f ~/Library/Logs/Opencode/ngrok.log
```

### 服务管理

```bash
# 停止服务
./mac-mini-deploy/stop-opencode.sh

# 启动服务
./mac-mini-deploy/start-opencode.sh

# 重启服务
./mac-mini-deploy/stop-opencode.sh
./mac-mini-deploy/start-opencode.sh
```

## 🧪 测试

### 单元测试

```bash
# 运行 ServerManager 测试
npm test -- ServerManager.test.ts
```

### 集成测试

```bash
# 1. 停止 Mac mini 服务
./mac-mini-deploy/stop-opencode.sh

# 2. 验证移动端自动切换到 Windows PC

# 3. 启动 Mac mini 服务
./mac-mini-deploy/start-opencode.sh

# 4. 验证移动端自动切换回 Mac mini
```

## ⚠️ 注意事项

### ngrok 免费版限制

1. **随机 URL**：每次重启可能变化
   - 解决：更新移动端配置
   - 或：升级 ngrok 付费版

2. **连接限制**：
   - 同时连接数有限
   - 可能需要定期重启隧道

3. **速度限制**：
   - 免费版带宽有限
   - 大量使用可能限速

### 网络稳定性

1. Mac mini 需要稳定的网络连接
2. 建议使用有线网络而非 WiFi
3. 配置网络断线自动重连

### 电力供应

1. 确保持续供电
2. 避免系统自动睡眠：
   ```bash
   sudo pmset -a disablesleep 1
   ```

## 🔒 安全建议

1. **访问控制**：
   - 考虑添加 API 密钥认证
   - 使用反向代理（nginx）

2. **HTTPS**：
   - ngrok 已提供 HTTPS
   - 生产环境建议使用自定义域名

3. **日志管理**：
   - 定期清理日志
   - 避免泄露敏感信息

## 📈 性能指标

### 预期性能

- **启动时间**：< 5 秒
- **健康检查**：< 1 秒
- **响应时间**：< 500ms（ngrok 隧道）
- **故障转移**：< 10 秒

### 资源使用

- **内存**：~200MB（OpenCode Server）
- **CPU**：~5%（空闲）
- **网络**：取决于使用频率

## 🎓 TDD 开发流程

本项目严格遵循 TDD 原则：

1. **RED**：先编写失败的测试
   - `ServerManager.test.ts` - 定义期望行为

2. **GREEN**：实现最小化代码让测试通过
   - `ServerManager.ts` - 实现功能

3. **REFACTOR**：重构优化代码
   - 提取配置
   - 优化结构
   - 改进命名

## 📚 相关文档

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - 详细部署指南
- [../PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - 项目总结
- [../src/services/ServerManager.ts](../src/services/ServerManager.ts) - 服务器管理器实现

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**作者**: Claude Sonnet 4.5
**日期**: 2026-02-09
**版本**: 1.0.0
