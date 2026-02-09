# Skills Mobile MVP - 设计文档

## 项目概述

**目标**：创建一个简单的React Native应用，通过手机访问OpenCode Server
**技术栈**：
- React Native 0.83.1
- TypeScript 5.3.3
- React Navigation 6.x
- Android SDK API 34

---

## 文件结构要求

### 1. 项目根目录结构

```
mobile_skills_mvp/
├── src/
│   ├── App.tsx                    # 应用入口组件
│   ├── index.ts                    # 注册入口
│   ├── app.json                     # 应用配置
│   ├── screens/
│   │   └── ChatScreen.tsx          # 聊天界面
│   ├── components/
│   │   ├── MessageBubble.tsx         # 消息气泡
│   │   ├── ChatInput.tsx             # 输入框
│   │   └── LoadingSpinner.tsx       # 加载指示器
│   └── services/
│       ├── openCode.ts                # 完整API服务
│       └── openCodeSimple.ts         # 简化API服务
├── android/
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/com/skillsmobile/app/
│   │   │   │   ├── MainActivity.java     # 主Activity
│   │   │   │   └── MainApplication.java # 应用入口
│   │   │   ├── AndroidManifest.xml      # 清单文件
│   │   │   └── res/                  # 资源文件
│   │   │       ├── values/
│   │   │       │   ├── strings.xml
│   │   │       │   └── styles.xml
│   │   │       ├── mipmap-*/        # 应用图标
│   │   │       └── drawable/
│   │   └── build.gradle             # 应用级构建配置
│   ├── build.gradle                 # 项目级构建配置
│   ├── settings.gradle              # Gradle设置
│   └── gradle.properties           # Gradle属性
├── package.json                   # 依赖配置
├── tsconfig.json                 # TypeScript配置
└── README.md                     # 项目文档
```

---

## 2. React Native源代码要求

### 2.1 入口文件 (src/index.ts)

```typescript
/**
 * @format
 */

import {AppRegistry} from 'react-native';
import App from './App';
import {name as appName} from './app.json';

AppRegistry.registerComponent(appName, () => App);
```

**要求**：
- ✅ 导入AppRegistry
- ✅ 导入App组件
- ✅ 导入app.json的name
- ✅ 注册组件

### 2.2 应用组件 (src/App.tsx)

```typescript
import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import ChatScreen from './screens/ChatScreen';

const Stack = createNativeStackNavigator();

export default function App(): JSX.Element {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen
          name="Chat"
          component={ChatScreen}
          options={{
            title: 'OpenCode Chat',
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

**要求**：
- ✅ 导入React
- ✅ 导入NavigationContainer
- ✅ 导入createNativeStackNavigator
- ✅ 导入ChatScreen
- ✅ 创建Stack实例
- ✅ 使用NavigationContainer包装
- ✅ 使用Stack.Navigator
- ✅ 使用Stack.Screen

### 2.3 应用配置 (src/app.json)

```json
{
  "name": "SkillsMobile",
  "displayName": "SkillsMobile"
}
```

**要求**：
- ✅ name与MainActivity.getMainComponentName()返回值一致
- ✅ displayName可选

---

## 3. Android原生代码要求

### 3.1 AndroidManifest.xml

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.skillsmobile.app">

    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <application
      android:name=".MainApplication"
      android:label="@string/app_name"
      android:icon="@mipmap/ic_launcher"
      android:allowBackup="false"
      android:theme="@style/AppTheme"
      android:usesCleartextTraffic="true">

      <activity
        android:name=".MainActivity"
        android:label="@string/app_name"
        android:configChanges="keyboard|keyboardHidden|orientation|screenLayout|screenSize|smallestScreenSize|layoutDirection"
        android:windowSoftInputMode="adjustResize"
        android:exported="true">

        <intent-filter>
            <action android:name="android.intent.action.MAIN" />
            <category android:name="android.intent.category.LAUNCHER" />
        </intent-filter>
      </activity>
    </application>
</manifest>
```

**关键检查点**：
- ✅ `usesCleartextTraffic` (不是`usesCleartextTraffic`)
- ✅ package与MainActivity包名一致
- ✅ MainApplication类存在
- ✅ MainActivity类存在
- ✅ INTERNET权限已添加

### 3.2 MainActivity.java

```java
package com.skillsmobile.app;

import com.facebook.react.ReactActivity;
import com.facebook.react.ReactActivityDelegate;
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint;
import com.facebook.react.defaults.DefaultReactActivityDelegate;
import android.os.Bundle;

public class MainActivity extends ReactActivity {

  @Override
  protected String getMainComponentName() {
    return "SkillsMobile";
  }

  @Override
  protected ReactActivityDelegate createReactActivityDelegate() {
    return new DefaultReactActivityDelegate(
      this,
      getMainComponentName(),
      DefaultNewArchitectureEntryPoint.getFabricEnabled()
    );
  }
}
```

**关键检查点**：
- ✅ 返回值与app.json中的name一致
- ✅ 导入正确的React Native类
- ✅ 实现getMainComponentName()
- ✅ 实现createReactActivityDelegate()

### 3.3 MainApplication.java

```java
package com.skillsmobile.app;

import android.app.Application;
import com.facebook.react.ReactApplication;
import com.facebook.react.ReactNativeHost;
import com.facebook.soloader.SoLoader;

public class MainApplication extends Application implements ReactApplication {

  private final ReactNativeHost mReactNativeHost =
      new ReactNativeHost(this) {
        @Override
        public boolean getUseDeveloperSupport() {
          return true;
        }

        @Override
        protected String getJSMainModuleName() {
          return "index";
        }
      };

  @Override
  public ReactNativeHost getReactNativeHost() {
    return mReactNativeHost;
  }

  @Override
  public void onCreate() {
    super.onCreate();
    SoLoader.init(this, /* native exopackage */ false);
  }
}
```

**关键检查点**：
- ✅ 实现ReactApplication接口
- ✅ 创建ReactNativeHost
- ✅ 配置JS模块名为"index"
- ✅ 开启开发者支持
- ✅ 初始化SoLoader
- ✅ 实现getReactNativeHost()

### 3.4 资源文件

#### strings.xml
```xml
<resources>
    <string name="app_name">SkillsMobile</string>
</resources>
```

#### styles.xml
```xml
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <style name="AppTheme" parent="Theme.AppCompat.Light.NoActionBar">
        <item name="android:windowBackground">@android:color/white</item>
    </style>
</resources>
```

#### ic_launcher.xml (应用图标)
```xml
<?xml version="1.0" encoding="utf-8"?>
<vector xmlns:android="http://schemas.android.com/apk/res/android"
    android:width="108dp"
    android:height="108dp"
    android:viewportWidth="108"
    android:viewportHeight="108">
  <group android:scaleX="0.5"
      android:scaleY="0.5"
      android:translateX="27"
      android:translateY="27">
    <path
        android:fillColor="#007AFF"
        android:pathData="M0,0h108v108h-108z"/>
  </group>
</vector>
```

---

## 4. Gradle配置要求

### 4.1 settings.gradle

```gradle
rootProject.name = 'SkillsMobile'
include ':app'
includeBuild('../node_modules/@react-native/gradle-plugin')
```

**关键检查点**：
- ✅ rootProject.name与app.json一致
- ✅ 包含app模块
- ✅ 包含React Native gradle插件

### 4.2 build.gradle (项目级)

```gradle
buildscript {
    ext {
        buildToolsVersion = "34.0.0"
        minSdkVersion = 24
        compileSdkVersion = 34
        targetSdkVersion = 34
        ndkVersion = "23.1.7779620"
    }
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.1.0")
    }
}

allprojects {
    repositories {
        maven {
            url("$rootDir/../node_modules/react-native/android")
        }
        google()
        mavenCentral()
    }
}
```

**关键检查点**：
- ✅ minSdkVersion = 24 (React Native 0.83要求)
- ✅ compileSdkVersion = 34
- ✅ AGP版本8.1.0
- ✅ 包含React Native Maven仓库

### 4.3 app/build.gradle

```gradle
apply plugin: "com.android.application"

android {
    ndkVersion rootProject.ext.ndkVersion
    compileSdkVersion rootProject.ext.compileSdkVersion

    namespace "com.skillsmobile.app"
    defaultConfig {
        applicationId "com.skillsmobile.app"
        minSdkVersion rootProject.ext.minSdkVersion
        targetSdkVersion rootProject.ext.targetSdkVersion
        versionCode 1
        versionName "0.1.0"
    }

    signingConfigs {
        debug {
            storeFile file('debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }
    }
    buildTypes {
        debug {
            signingConfig signingConfigs.debug
        }
    }
}

dependencies {
    implementation("com.facebook.react:react-android:+")
    implementation("androidx.swiperefreshlayout:swiperefreshlayout:1.0.0")
}
```

**关键检查点**：
- ✅ 应用插件
- ✅ namespace与AndroidManifest包名一致
- ✅ applicationId与包名一致
- ✅ minSdkVersion = 24
- ✅ 依赖react-android（动态版本）

### 4.4 gradle.properties

```properties
org.gradle.jvmargs=-Xmx4g -XX:MaxPermSize=512k
android.useAndroidX=true
android.enableJetifier=true
org.gradle.java.home=D\:\\Soft\\Android\\jbr
```

---

## 5. 依赖检查

### package.json必需依赖

```json
{
  "name": "SkillsMobile",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "android": "react-native run-android"
  },
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.83.1",
    "axios": "^1.6.7",
    "@react-navigation/native": "^6.1.17",
    "@react-navigation/native-stack": "^6.1.17",
    "react-native-safe-area-context": "^4.10.1"
  },
  "devDependencies": {
    "@babel/core": "^7.24.0",
    "@babel/preset-env": "^7.24.0",
    "@babel/runtime": "^7.24.0",
    "typescript": "^5.3.3"
  }
}
```

---

## 6. 崩溃诊断清单

### 阶段1：安装问题
- [ ] APK是否成功安装（是否显示应用图标）
- [ ] 安装时是否有错误提示

### 阶段2：启动问题
- [ ] 点击图标是否立即闪退
- [ ] 是否有启动画面
- [ ] 闪退时机（立即、1秒后、3秒后）

### 阶段3：配置问题
- [ ] MainActivity.getMainComponentName()返回值是否正确
- [ ] app.json的name是否正确
- [ ] MainApplication是否正确实现
- [ ] React Native版本是否兼容

### 阶段4：权限问题
- [ ] INTERNET权限是否添加
- [ ] 网络访问是否被阻止

### 阶段5：资源问题
- [ ] AppTheme是否存在
- [ ] ic_launcher图标是否存在
- [ ] strings.xml是否存在

### 阶段6：代码问题
- [ ] Stack.Navigator是否正确导入
- [ ] NavigationContainer是否正确使用
- [ ] App.tsx是否正确导出

---

## 7. 调试方法

### 7.1 使用adb获取崩溃日志

```cmd
# 连接设备
adb devices

# 清除旧日志
adb logcat -c

# 启动应用并捕获日志
adb shell am start -n com.skillsmobile.app/.MainActivity
adb logcat -d *:E AndroidRuntime:E > crash.txt

# 查看崩溃信息
type crash.txt | findstr "FATAL"
```

### 7.2 使用手机自带工具（小米12s）

1. 进入开发者模式
   - 设置 → 我的设备 → 全部参数信息
   - 连续点击"MIUI版本"7次

2. 查看错误报告
   - 设置 → 更多设置 → 开发者选项
   - 点击"错误报告"

3. 复制错误报告内容

---

## 8. 下一步行动

### 8.1 一次性检查所有文件
- [ ] 检查AndroidManifest.xml的usesCleartextTraffic拼写
- [ ] 检查MainActivity.java的返回值
- [ ] 检查app.json的name
- [ ] 检查MainApplication.java的getJSMainModuleName
- [ ] 检查App.tsx的导入和组件
- [ ] 检查所有资源文件是否存在

### 8.2 修复所有发现问题
- [ ] 修复拼写错误
- [ ] 修复导入错误
- [ ] 修复配置错误

### 8.3 重新构建和测试
- [ ] 清理构建缓存
- [ ] 重新构建APK
- [ ] 安装并测试

---

## 9. 成功标准

应用成功启动的标准：
1. ✅ APK可以安装
2. ✅ 应用图标显示在手机上
3. ✅ 点击图标不闪退
4. ✅ 显示聊天界面
5. ✅ 可以输入消息
6. ✅ 可以发送消息

---

## 10. 调试方法

### 10.1 使用adb获取崩溃日志（最简单）

```cmd
# 连接设备
adb devices

# 清除旧日志
adb logcat -c

# 启动应用并捕获日志
adb shell am start -n com.skillsmobile.app/.MainActivity
adb logcat -d *:E AndroidRuntime:E > crash.txt

# 查看崩溃信息
type crash.txt | findstr "FATAL"
```

### 10.2 使用手机自带工具（小米12s）

1. **进入开发者模式**：
   - 设置 → 我的设备 → 全部参数信息
   - 连续点击"MIUI版本"7次
   
2. **使用Bug报告**：
   - 设置 → 更多设置 → 开发者选项
   - 点击"错误报告"

3. **复制错误报告**：
   - 长按应用 → Bug报告 → 复制日志

---

## 11. 最终APK信息

**文件名**: `SkillsMobile-v0.1.2-FINAL.apk`
**位置**: `D:\AI\2026\LearningSystem\SkillsMobile-v0.1.2-FINAL.apk`
**大小**: 85 MB
**版本**: v0.1.2 (最终修复版）

### 修复内容：
- ✅ 修复AndroidManifest.xml的`usesCleartextTraffic`拼写错误
- ✅ 检查所有文件配置正确性
- ✅ 一次性完成所有检查

---

## 12. 已知问题记录

| 问题 | 状态 | 解决方案 |
|------|------|----------|
| AndroidManifest拼写错误 | ✅ 已修复 | usesCleartextTraffic (修复usesCleartext拼写） |
| MainActivity返回值 | ✅ 已检查 | 返回"SkillsMobile" |
| MainApplication模块名 | ✅ 已检查 | 返回"index" |
| app.json配置 | ✅ 已检查 | name: "SkillsMobile" |
| index.ts注册 | ✅ 已检查 | 注册SkillsMobile |
| App.tsx导入 | ✅ 已检查 | 所有导入正确 |
| 资源文件 | ✅ 已检查 | 全部存在 |
| Gradle配置 | ✅ 已检查 | 全部正确 |
