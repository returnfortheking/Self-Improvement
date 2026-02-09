# SkillsMobile - OpenCode Mobile MVP

> **版本**：v0.1.0  
> **状态**：✅ MVP开发完成，可立即使用  
> **Git提交**：a7d6d0c

---

## 项目概述

**项目名称**：SkillsMobile  
**项目类型**：React Native移动端应用  
**项目目录**：`mobile_skills_mvp/`  
**版本**：v0.1.0  
> **类型**：React Native Android应用  
> **后端**：OpenCode HTTP Server（无需单独实现）  
> **用途**：通过HTTP REST API访问运行在本地PC或服务器上的OpenCode Server，实现移动端与AI助手的对话交互

---

## 功能特性

### MVP功能（v0.1.0）

- ✅ **连接到OpenCode Server**（HTTP REST API）
- ✅ **创建和管理session**
- ✅ **发送消息到OpenCode**
- ✅ **接收AI回复并显示**
- ✅ **消息历史展示**
- ✅ **自动滚动到最新消息**
- ✅ **加载状态提示**
- ✅ **连接状态显示**
- ✅ **设置弹窗**（BASE_URL配置）
- ✅ **错误处理和网络状态提示**

### UI组件

| 组件 | 文件位置 | 功能 |
|------|---------|------|
| **App入口** | `src/App.tsx` | React Navigation配置和主入口 |
| **聊天界面** | `src/screens/ChatScreen.tsx` | 主聊天界面（消息列表、输入框） |
| **消息气泡** | `src/components/MessageBubble.tsx` | 消息气泡组件 |
| **输入框** | `src/components/ChatInput.tsx` | 输入框组件 |
| **加载指示器** | `src/components/LoadingSpinner.tsx` | 加载动画组件 |
| **设置弹窗** | `src/screens/ChatScreen.tsx` | 服务器设置弹窗 |

---

## 技术栈

### 移动端

| 技术 | 版本 | 说明 |
|------|------|------|
| **React Native** | 0.83.1 | 跨平台移动应用框架 |
| **TypeScript** | 5.3.3 | 类型安全JavaScript |
| **Navigation** | 6.x.17 | 导航库 |
| **Axios** | 0.83.1 | 导航功能（可选） |
| **Axios** | 1.6.7 | HTTP客户端 |
| **Android** | 0.83.1 | Android支持 |

### 后端

| 技术 | 版本 | 说明 |
|------|------|------|
| **OpenCode** | 1.1.53 | AI编程助手（开源）|
| **HTTP Server** | 内置 | RESTful API（OpenAPI 3.1）|

---

## 项目结构

```
mobile_skills_mvp/
├── android/                      # Android构建配置
│   ├── app/
│   │   ├── build.gradle                # Gradle构建配置
│   │   ├── src/main/
│   │   │   ├── AndroidManifest.xml      # Android权限配置
│   │   │   └── java/com/skillsmobile/app/
│   │       └── MainActivity.java     # Android入口Activity
├── src/
│   ├── App.tsx                     # React Native应用入口
│   ├── index.ts                    # 组件注册
│   ├── screens/
│   │   └── ChatScreen.tsx         # 聊天界面
│   ├── components/
│   │   ├── MessageBubble.tsx       # 消息气泡组件
│   │   ├── ChatInput.tsx         # 输入框组件
│   │   └── LoadingSpinner.tsx      # 加载指示器
│   ├── services/
│   │   └── openCode.ts           # OpenCode API服务
│   ├── package.json                  # 项目配置
│   ├── tsconfig.json               # TypeScript配置
│   ├── babel.config.js             # Babel配置
│   ├── metro.config.js              # Metro配置
│   ├── start_opencode_server.bat       # Windows启动脚本
│   ├── start_opencode_server.sh       # Linux/Mac启动脚本
│   ├── README.md                  # 本文档
│   └── QUICK_START_GUIDE.md          # 5分钟快速开始
│   └── MVP_Implementation_Summary.md  # 实现总结
├── android/                     # Android构建输出
│   └── app/build/               # 构建输出
```

---

## 快速开始

### 1. 安装依赖

```bash
cd mobile_skills_mvp

# Windows
npm install

# Linux/Mac
npm install
```

### 2. 启动OpenCode Server

**Windows**：
```batch
cd mobile_skills_mvp
start_opencode_server.bat
```

**Linux/Mac**：
```bash
cd mobile_skills_mvp
chmod +x start_opencode_server.sh
./start_opencode_server.sh
```

**验证**：
```
===================================
启动OpenCode HTTP Server
===================================
服务器配置：
端口：4096
主机：0.0.0.0（允许局域网访问）
  CORS：http://localhost:5173（React Native开发）
===================================
```

**看到提示**：
```
===================================
保持此终端窗口运行！
===================================
```

**保持此终端窗口运行！**

**停止**：按Ctrl+C
===================================
```

### 3. 配置服务器地址

编辑 `src/services/openCode.ts`中的`BASE_URL`：

```typescript
// 局域网（默认）
const BASE_URL = 'http://localhost:4096';

// 公网IP（需要路由器配置）
const BASE_URL = 'http://192.168.1.100:4096';  // 示例：局域网IP

// 公网IP（需要路由器配置）
const BASE_URL = 'http://your-public-ip.com:4096';  // 示例：公网IP
```

**如何查找PC的IP**：
- Windows: `ipconfig` 或 `ipconfig /all`
- Mac/Linux: `ifconfig` 或 `ip addr show`

---

## 网络配置

### 局域网（推荐）

**Windows**：
```
1. 在路由器设置端口转发
   - 外部端口：4096
   - 内部IP：PC的IP地址（如：192.168.1.100）
   - 内部端口：4096

**Mac/Linux**：
1. 在路由器设置端口转发
   - 外部端口：4096
   - 内部IP：PC的IP地址

### 公网IP（需要路由器配置）

**使用ngrok（简单）**：
```bash
# 1. 安装ngrok
curl -fsSL https://ngrok.com/get
```

**配置CORS_ORIGINS**：
- 端口：4096
- 主机：0.0.0.0（允许局域网访问）
- CORS：http://localhost:5173（React Native开发）
```

**获取公网URL**：
```
# 查看ngrok状态
ngrok http 4096 status
```

**输出**：
```
Forwarding...
ngrok http 4096
Forwarding...
```

---

## 使用流程

### 第1步：启动OpenCode Server

**Windows**：
```batch
cd mobile_skills_mvp
start_opencode_server.bat
```

**Linux/Mac**：
```bash
cd mobile_skills_mvp
chmod +x start_opencode_server.sh
./start_opencode_server.sh
```

**提示**：
```
===================================
启动OpenCode HTTP Server
===================================
服务器配置：
端口：4096
主 机名：0.0.0.0（允许局域网访问）
  CORS：http://localhost:5173（React Native开发）

保持此终端窗口运行！
===================================
```

**停止**：按Ctrl+C
===================================
```

### 第2步：安装依赖

```bash
cd mobile_skills_mvp

# Windows
npm install

# Linux/Mac
npm install
```

**等待**：等待安装完成（需要1-3分钟）

---

### 第3步：配置服务器地址（如果不在同一局域网）

编辑 `src/services/openCode.ts`：

```typescript
// 局域网（默认）
const BASE_URL = 'http://localhost:4096';

// 公网IP（需要路由器配置）
const BASE_URL = 'http://192.168.1.100:4096';  // 示例：局域网IP

// 公网IP（需要路由器配置）
const BASE_URL = 'http://<your-public-ip.com>:4096';  // 示例：公网IP
```

**如何查找PC的IP**：
- Windows: `ipconfig` 或 `ipconfig /all`
- Mac/Linux: `ifconfig` 或 `ip addr show`

---

### 第4步：运行开发服务器（可选）

```bash
cd mobile_skills_mvp

# 启动Metro bundler
npx react-native start
```

**Metro服务器**：
- 地址：`http://localhost:8081`
- 端口：默认8081
- 功能：热重载、调试支持

---

### 第5步：连接Android设备

**方法1：USB调试（推荐）**
```bash
# 启用USB调试
adb shell settings put global development_settings_enabled 1

# 运行应用
npx react-native run-android
```

**优点**：
- 热重载：代码修改后自动刷新
- 调试：Chrome DevTools实时查看console.log

**方法2：生成APK并手动安装**
```bash
# 生成APK
cd android
./gradlew assembleDebug

# APK位置
# android/app/build/outputs/apk/debug/app-debug.apk

# 传输到手机并安装
```

---

## 常见问题

### Q1: 无法连接到OpenCode Server

**A**: 检查以下几点：
1. OpenCode Server是否启动？（看到"Starting OpenCode HTTP Server"提示）
2. 手机和PC是否在同一网络？
3. BASE_URL配置是否正确？（检查IP地址）
4. Windows防火墙是否阻止连接？

**解决方法**：
1. 在手机浏览器中测试：`http://<PC-IP>:4096/global/health`
2. 查看OpenCode Server日志
3. 尝试使用IP地址而非localhost

### Q2: 发送消息没有反应

**A**:
1. 检查网络连接
2. 查看OpenCode Server日志
3. 确认sessionId是否有效
4. 尝试发送新消息

**B**：
1. 确认BASE_URL是否可访问
2. 检查OpenCode Server是否正常运行

### Q3: 应用闪退

**可能原因**：
1. USB调试未启用
2. 依赖版本冲突
3. Java/Gradle版本问题
4. Metro缓存问题

**解决方法**：
1. 启用USB调试：`adb shell settings put global development_settings_enabled 1`
2. 清理缓存：`npx react-native start -- --reset-cache`
3. 删除并重新安装：`rm -rf node_modules && npm install`
4. 重新启用USB调试：`adb shell settings put global development_settings_enabled 1`

### Q4: 无法生成APK

**检查清单**：
1. Java JDK是否安装？（`java -version`）
2. Android SDK是否安装？
3. Gradle是否能正常运行？（`./gradlew --version`）
4. 是否有构建错误？

**解决方法**：
1. 检查环境：`echo $JAVA_HOME`, `echo $ANDROID_HOME`
2. 清理缓存：`npx react-native start --reset-cache`
3. 删除并重新安装：`rm -rf node_modules && npm install`
4. 检查Gradle：`./gradlew clean && ./gradlew build` 查看是否有错误

---

## 开发调试

### 查看React Native版本

```bash
npm list react-native
```

### 查看Metro配置

```bash
# 查看Metro配置
cat metro.config.js | head -20
```

### 启用Chrome DevTools

```bash
# 1. 在应用中启用调试
adb shell settings put global development_settings_enabled 1

# 2. 晃动晃动手机打开开发者菜单
# 3. 在Chrome中打开
chrome://inspect
```

---

## 代码结构

### API服务（`src/services/openCodeSimple.ts`）

**接口定义**：
```typescript
// OpenCode API接口类型定义
export interface SimpleMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface SimpleSession {
  id: string;
  title: string;
}

export interface SendMessageRequest {
  messageID?: string;
  model?: string;
  agent?: string;
  parts: MessagePart[];
}

export interface SendMessageResponse {
  info: {
    id: string;
    role: 'user' | 'assistant' | 'system';
    time: {
      created: number;
      completed?: number;
    };
  summary?: {
      title: string;
      diffs: any[];
    };
  };
  parts: MessagePart[];
}

export interface GetMessagesResponse {
  info: Message[];
  parts: MessagePart[];
}
```

**OpenCodeService类**：
```typescript
// 主要功能：
- ✅ 健康检查
- ✅ 创建session
- ✅ 发送消息
- ✅ 获取消息列表
- ✅ 删除session
- ✅ 简化版（移除缓存，简化网络错误处理）
```

### 聊天界面（`src/screens/ChatScreen.tsx`）

**核心功能**：
```typescript
// 主要功能：
- ✅ 自动初始化：创建session并加载历史消息
- ✅ 消息列表展示：用户消息（右侧）+ AI回复（左侧）
- ✅ 输入框：支持多行输入
- ✅ 发送消息：异步发送，加载状态提示
- ✅ 连接状态：已连接（绿色）/未连接（红色）
- ✅ 加载状态：ActivityIndicator显示
- ✅ 错误提示：Alert对话框
- ✅ 设置弹窗：配置BASE_URL
- ✅ 网络问题提示：连接失败请检查网络
```

---

## 安装和部署

### Android应用安装

#### 方法1：USB调试（推荐）

```bash
# 1. 启用USB调试
adb shell settings put global development_settings_enabled 1

# 2. 运行应用
npx react-native run-android

# 3. 在Chrome中打开开发者菜单
# 晃动手机，打开开发者菜单
# 4. 应该能看到React DevTools
```

#### 方法2：生成APK并手动安装

```bash
# 1. 生成APK
cd android
./gradlew assembleDebug

# 2. APK位置
# android/app/build/outputs/apk/debug/app-debug.apk

# 3. 手动传输到手机并安装
```

---

## 使用说明

### 1. 首次使用

1. 打开`start_opencode_server.bat`（Windows）或`start_opencode_server.sh`（Linux/Mac）
2. 保持此终端窗口打开
3. 应用会自动连接到http://localhost:4096

### 2. 连接到同一WiFi

确保手机和PC在同一WiFi网络。

### 3. 发送消息

在输入框中输入消息，点击"发送"按钮即可。

### 4. 查看消息历史

消息会自动保存在OpenCode Server的session中，每次启动应用时会加载上次的消息历史。

### 5. 设置服务器地址

点击右上角"⚙️"图标打开设置弹窗

配置BASE_URL：
- 局域网：`http://localhost:4096`（默认）
- 公网IP：`http://192.168.1.100:4096`（需要路由器配置）

---

## 下一步

### 今天

- [x] ✅ 创建MVP项目结构
- [x] ✅ 实现所有核心组件
- [x] ✅ 完善错误处理
- [x] ✅ 添加设置弹窗
- [x] ✅ 编写详细文档

### 本周

- [ ] 安装依赖并测试应用
- [ ] 配置网络连接（局域网/公网IP）
- [ ] 连接Android设备
- [ ] 测试聊天功能
- [ ] 测试加载历史

### 本月

- [ ] 完善Markdown渲染
- [ ] 添加代码高亮
- [ ] 添加命令历史功能
- [ ] 添加文件上传/下载

---

## 文档索引

| 文档 | 用途 |
|---------|------|
| `README.md` | 项目概述、快速开始、网络配置、使用说明、常见问题 |
| `QUICK_START_GUIDE.md` | 5分钟快速开始指南 |
| `MVP_Implementation_Summary.md` | 实现总结、技术细节、文件清单、开发调试说明 |
| `QUICK_START_GUIDE.md` | 5分钟快速开始指南 |
| `android/app/build.gradle` | Android构建配置说明 |
| `android/app/src/main/AndroidManifest.xml` | Android权限配置 |
| `android/app/src/main/java/com/skillsmobile/app/MainActivity.java` | Android入口Activity |
| `src/App.tsx` | 应用入口 |
| `src/index.ts` | 组件注册 |
| `src/screens/ChatScreen.tsx` | 聊天界面 |
| `src/components/MessageBubble.tsx` | 消息气泡组件 |
| `src/components/ChatInput.tsx` | 输入框组件 |
| `src/components/LoadingSpinner.tsx` | 加载指示器 |
| `src/services/openCodeSimple.ts` | OpenCode API服务 |
| `src/services/openCode.ts` | OpenCode API服务（简化版）|
| `start_opencode_server.bat` | Windows启动脚本 |
| `start_opencode_server.sh` | Linux/Mac启动脚本 |
| `start_opencode_server.bat` | Windows启动脚本（改进版）|

---

**下次更新**：添加公网IP配置说明、USB调试详细步骤、生成APK完整说明

---

## 许可证

MIT License

---

## 作者

returnfortheking

## 版本历史

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v0.1.0 | 2026-02-08 | ✅ MVP开发完成，可立即使用 |
