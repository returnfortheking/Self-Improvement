# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**SkillsMobile** 是一个 React Native Android 应用，通过 HTTP REST API 连接到 OpenCode Server（AI 编程助手），实现移动端与 AI 助手的对话交互。

### 架构模式

- **客户端-服务器架构**：移动端通过 HTTP 与本地或远程的 OpenCode Server 通信
- **Session 管理**：使用 OpenCode Server 的 session 机制维护对话上下文
- **简化 API 服务**：`openCodeSimple.ts` 提供简化的 API 接口，包含本地缓存以减少网络请求

### 核心依赖

- React Native 0.83.1
- TypeScript 5.3.3
- React Navigation 6.x
- Axios 1.6.7
- Android SDK API 34 (minSdkVersion: 24)

## 常用命令

### 开发命令

```bash
# 安装依赖
npm install

# 启动 Metro bundler（开发服务器）
npm start

# 运行 Android 应用（需要连接设备或模拟器）
npm run android

# 运行测试
npm test
```

### Android 构建命令

```bash
# 生成 Debug APK
cd android
./gradlew assembleDebug

# APK 输出位置
# android/app/build/outputs/apk/debug/app-debug.apk

# 清理构建缓存
./gradlew clean

# 通过 USB 直接安装到设备
npx react-native run-android
```

### OpenCode Server 命令

```bash
# Windows
start_opencode_server.bat

# Linux/Mac
chmod +x start_opencode_server.sh
./start_opencode_server.sh
```

Server 默认配置：
- 端口：4096
- 主机：0.0.0.0（允许局域网访问）
- CORS：http://localhost:5173

## 项目架构

### 目录结构

```
src/
├── App.tsx                      # 应用入口，React Navigation 配置
├── index.ts                     # 组件注册入口
├── screens/
│   └── ChatScreen.tsx          # 聊天界面（主屏幕）
├── components/
│   ├── MessageBubble.tsx       # 消息气泡组件
│   ├── ChatInput.tsx           # 输入框组件
│   └── LoadingSpinner.tsx      # 加载动画组件
└── services/
    └── openCodeSimple.ts       # OpenCode API 服务（简化版）
```

### 数据流

1. **初始化流程**：ChatScreen 启动时自动创建 session 并加载历史消息
2. **发送消息**：用户输入 → ChatInput → sendMessage API → OpenCode Server → 更新消息列表
3. **接收回复**：Server 返回响应 → 更新消息状态 → 自动滚动到最新消息

### API 服务设计

`OpenCodeServiceSimple` 类（[openCodeSimple.ts](src/services/openCodeSimple.ts)）：

- 使用单例模式导出默认实例
- 内置缓存机制减少网络请求（healthCheck、sessionInfo、messages）
- 简化的错误处理，依赖 axios 的默认超时和重试
- 核心方法：
  - `healthCheck()`: 检查 Server 健康状态
  - `createSession(title)`: 创建新对话 session
  - `sendMessage(sessionId, content)`: 发送消息
  - `getMessages(sessionId)`: 获取消息历史
  - `deleteSession(sessionId)`: 删除 session

### 组件关系

```
App.tsx (NavigationContainer)
  └── Stack.Navigator
      └── ChatScreen (Stack.Screen)
          ├── MessageBubble (FlatList 渲染)
          ├── ChatInput (用户输入)
          └── LoadingSpinner (加载状态)
```

## 重要配置

### Server 地址配置

编辑 [src/services/openCodeSimple.ts](src/services/openCodeSimple.ts:8) 中的 `BASE_URL`：

```typescript
// 默认本地开发
const BASE_URL = 'http://localhost:4096';

// 局域网访问（使用 PC 的 IP 地址）
const BASE_URL = 'http://192.168.1.100:4096';
```

### Android 网络权限

[AndroidManifest.xml](android/app/src/main/AndroidManifest.xml:152) 必须包含：
- `INTERNET` 权限
- `usesCleartextTraffic="true"`（允许 HTTP 连接）

### 应用名称一致性

以下文件中的应用名称必须保持一致：
- [app.json](app.json): `name` 字段
- [MainActivity.java](android/app/src/main/java/com/skillsmobile/app/MainActivity.java:192): `getMainComponentName()` 返回值
- [index.ts](src/index.ts): `AppRegistry.registerComponent()` 第一个参数

当前应用名称：`SkillsMobile`

## 开发注意事项

### 网络调试

- 使用手机浏览器测试 Server 连接：`http://<PC-IP>:4096/global/health`
- 确保手机和 PC 在同一 WiFi 网络
- Windows 防火墙可能阻止连接，需要允许端口 4096

### Metro 缓存问题

如果遇到模块加载错误：
```bash
npx react-native start -- --reset-cache
```

### USB 调试

启用 USB 调试：
```bash
adb shell settings put global development_settings_enabled 1
```

### Chrome DevTools 调试

1. 在应用中晃动手机打开开发者菜单
2. 选择 "Debug"
3. 在 Chrome 中访问 `chrome://inspect`

## 技术约束

- **目标平台**：仅 Android（iOS 未配置）
- **最低 Android 版本**：API 24 (Android 7.0)
- **编译 SDK 版本**：API 34
- **Gradle 插件版本**：8.1.0
- **Node 版本要求**：>= 18
- **npm 版本要求**：>= 9

## 相关文档

- [README.md](README.md)：项目概述、快速开始、常见问题
- [MVP_DESIGN_DOCUMENT.md](MVP_DESIGN_DOCUMENT.md)：详细设计文档和配置说明
- [CRASH_FIX_REPORT.md](CRASH_FIX_REPORT.md)：崩溃修复记录
- [AGENTS.md](AGENTS.md)：AI Agent 相关说明
