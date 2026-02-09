# SkillsMobile MVP - 快速使用指南

> **版本**：v0.1.0  
> **状态**：✅ MVP开发完成，可立即使用

---

## 🚀 5分钟快速开始

### 步骤1：启动OpenCode Server（PC端）

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

**看到提示**：
```
===================================
启动OpenCode HTTP Server
===================================
服务器配置：
  端口：4096
  主机名：0.0.0.0（允许局域网访问）
  CORS：http://localhost:5173（React Native开发）

使用方法：
 1. 保持此终端窗口打开
  2. 在React Native应用中配置BASE_URL为http://<你的IP>:4096
  3. 如果使用手机和PC在同一WiFi，使用PC的IP地址

停止：按Ctrl+C
===================================
```

**保持此终端窗口运行！**

---

### 步骤2：安装依赖（第一次运行时）

```bash
cd mobile_skills_mvp

# Windows
npm install

# Linux/Mac
npm install
```

**等待安装完成**（需要1-3分钟）

---

### 步骤3：运行开发服务器（可选）

```bash
cd mobile_skills_mvp
npx react-native start
```

**Metro服务器**：
- 地址：`http://localhost:8081`
- 功能：热重载、调试支持

**可选**：如果你有Android Studio，可以直接在Android Studio中运行

---

### 步骤4：配置服务器地址（如果不在同一局域网）

编辑 `src/services/openCode.ts`：
```typescript
// 开发环境（默认）
const BASE_URL = 'http://localhost:4096';

// 生产环境（使用PC的局域网IP或公网IP）
const BASE_URL = 'http://192.168.1.100:4096';  // 示例
const BASE_URL = 'http://<你的公网IP>:4096';     // 需要路由器配置
```

**如何查找PC的IP**：
- Windows: `ipconfig` 或 `ipconfig /all`
- Mac/Linux: `ifconfig` 或 `ip addr show`

---

### 步骤5：运行APK（开发阶段）

#### 方法1：USB调试（推荐）

```bash
# 1. 启用USB调试
adb shell settings put global development_settings_enabled 1

# 2. 运行应用
npx react-native run-android
```

**优点**：
- ✅ 热重载：代码修改后自动刷新
- ✅ 调试：Chrome DevTools
- ✅ 日志：实时查看console.log

#### 方法2：生成APK并手动安装

```bash
# 1. 生成APK
cd android
./gradlew assembleDebug

# 2. APK位置
# android/app/build/outputs/apk/debug/app-debug.apk

# 3. 手动传输到手机并安装
```

**APK文件位置**：
```
mobile_skills_mvp/android/app/build/outputs/apk/debug/app-debug.apk
```

---

## 📱 使用说明

### 首次打开应用

1. 应用会自动创建session
2. 连接状态显示"已连接"（绿色）
3. 等待你输入消息

### 发送消息

1. 在输入框中输入消息
2. 点击"发送"按钮
3. 消息会显示在聊天界面（右侧）
4. AI回复会显示在聊天界面（左侧）
5. 自动滚动到最新消息

### 查看消息历史

消息会自动保存在OpenCode Server的session中  
当前session的消息会一直显示在界面上

### 停止连接

如果OpenCode Server停止：
1. 连接状态会变成"未连接"（红色）
2. 无法发送新消息
3. 需要重新启动OpenCode Server

---

## 🔧 配置和故障排除

### 问题1：无法连接到OpenCode Server

**检查清单**：
- [ ] OpenCode Server是否启动？（看到"Starting OpenCode HTTP Server"提示）
- [ ] PC和手机是否在同一WiFi？
- [ ] BASE_URL配置是否正确？（检查IP地址）
- [ ] Windows防火墙是否阻止连接？
- [ ] 网络是否通畅？

**解决方法**：
1. 在手机浏览器中测试：`http://<PC-IP>:4096/global/health`
2. 检查OpenCode Server日志
3. 尝试使用IP地址而非localhost

### 问题2：应用闪退

**可能原因**：
1. USB调试未启用
2. 依赖版本冲突
3. Java/Gradle版本问题
4. Metro缓存问题

**解决方法**：
```bash
# 1. 清理缓存
npx react-native start --reset-cache

# 2. 删除并重新安装
rm -rf node_modules
npm install

# 3. 重新启用USB调试
adb shell settings put global development_settings_enabled 1
```

### 问题3：看不到console.log

**解决方法**：
```bash
# 1. 在应用中启用调试
# 2. 晃动手机打开开发者菜单
# 3. 在Chrome中打开：chrome://inspect
# 4. 应该能看到React DevTools
```

### 问题4：无法生成APK

**检查清单**：
- [ ] Java JDK是否安装？（`java -version`）
- [ ] Android SDK是否安装？
- [ ] Gradle是否能正常运行？（`./gradlew --version`）
- [ ] 是否有构建错误？

**解决方法**：
```bash
# 1. 检查环境
java -version
./gradlew --version

# 2. 清理缓存
npx react-native start --reset-cache

# 3. 清理并重新安装
rm -rf node_modules
npm install

# 4. 重新构建
./gradlew clean
./gradlew assembleDebug
```

---

## 📊 网络配置

### 局域网（推荐用于开发）

**PC IP查看**：
```batch
# Windows
ipconfig
# Linux/Mac
ifconfig
```

**手机连接同一WiFi**：
1. 确保手机和PC连接到同一个路由器
2. 手机WiFi设置中查看PC的IP
3. 配置BASE_URL

### 公网IP（生产环境，需要路由器配置）

**使用ngrok（简单）**：
```bash
# 1. 安装ngrok
# 2. 注册账号并获取authtoken
# 3. 运行ngrok
ngrok http 4096

# 输出：Forwarding...
# 4. 获得公网URL，如：https://abc123.ngrok.io
```

**配置BASE_URL**：
```typescript
const BASE_URL = 'https://abc123.ngrok.io:4096';
```

**使用frp（更稳定）**：
```bash
# 1. 配置frp服务器
# 2. 启动frp
frp start-server --region=us --http 4096

# 3. 获得公网域名，如：yourname.frp.dev
```

**配置BASE_URL**：
```typescript
const BASE_URL = 'http://yourname.frp.dev:4096';
```

### 路由器端口转发

**TP-Link / AX881**（推荐）：
1. 登录路由器管理页面
2. 外部端口：4096
3. 内部IP：PC的IP地址
4. 内部端口：4096

**其他路由器**：
- 华为：Virtual Server → 端口映射
- 小米：端口映射
- 联通：端口映射
- Google WiFi：端口映射

---

## 🎯 下一步

### 今天

- [ ] 安装依赖
- [ ] 启动OpenCode Server测试
- [ ] 配置网络连接
- [ ] 运行开发服务器
- [ ] 连接Android设备测试

### 本周

- [ ] 完成功能测试
- [ ] 解决网络连接问题
- [ ] 优化UI体验
- [ ] 添加Markdown渲染
- [ ] 添加代码高亮

---

## 📞 联系和支持

### 文档

- **项目README**：`mobile_skills_mvp/README.md`
- **OpenCode API文档**：`http://localhost:4096/doc`
- **React Native文档**：`https://reactnative.dev/`

### 项目状态

- ✅ **MVP开发完成**：12个核心文件
- ✅ **技术方案验证**：OpenCode HTTP Server完全可行
- ⏳ **测试阶段**：待安装依赖和测试
- ⏳ **部署阶段**：待配置网络和打包

---

**下一步**：安装依赖并开始测试
