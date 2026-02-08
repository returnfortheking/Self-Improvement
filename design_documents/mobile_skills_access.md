# Skills系统移动端访问 - 设计文档

> **版本**: v0.1 (初始版本)
> **创建日期**: 2026-02-08
> **状态**: 设计阶段
> **维护者**: returnfortheking

---

## 📋 文档更新记录

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v0.2 | 2026-02-08 | **重大更新**：验证OpenCode HTTP Server可行性，改用REST API架构 | returnfortheking |
| v0.1 | 2026-02-08 | 初始版本，明确核心需求和架构 | returnfortheking |

---

## 📌 核心需求

### 问题陈述

**当前系统**：
- Skills v3.0系统运行在本地PC/服务器
- 使用Claude Code或Opencode CLI进行交互
- 只能在安装了CLI的设备上使用

**用户需求**：
- 希望通过手机访问Skills系统
- 在手机上看到CLI的输出
- 在手机上输入命令给CLI

### 核心挑战

1. **如何获取CLI的输出（stdout/stderr）并传输给手机端？**
2. **如何将手机端的输入传输给CLI的stdin？**

---

## 🏗️ 技术架构

### ✅ 实验验证结果

**2026-02-08实验验证成功！**

验证命令：
```bash
# 启动OpenCode HTTP服务器
opencode serve --port 4096 --hostname 0.0.0.0

# 健康检查
curl http://localhost:4096/global/health
# 返回: {"healthy":true,"version":"1.1.53"}

# 创建session
curl -X POST http://localhost:4096/session -H "Content-Type: application/json" -d '{"title":"测试会话"}'

# 发送消息
curl -X POST http://localhost:4096/session/{session_id}/message \
  -H "Content-Type: application/json" \
  -d '{"parts":[{"type":"text","text":"Hello"}]}'

# 获取消息列表
curl http://localhost:4096/session/{session_id}/message
```

**结论**：✅ OpenCode HTTP Server完全可行！REST API稳定可靠！

---

### 整体架构图（v0.2更新）

```
┌─────────────────────────────────────────────────────────┐
│                   React Native 移动端                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  会话UI                                             │  │
│  │  - 显示消息历史                                     │  │
│  │  - 发送用户消息                                     │  │
│  │  - HTTP客户端                                       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           │ HTTP REST API
                           │
┌──────────────────────────▼──────────────────────────────┐
│           OpenCode HTTP Server (opencode serve)        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  RESTful API (OpenAPI 3.1)                      │  │
│  │  - Session管理                                      │  │
│  │  - Message发送                                      │  │
│  │  - 消息历史查询                                    │  │
│  │  - Files/Tools访问                                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           │ 内部调用
                           │
┌──────────────────────────▼──────────────────────────────┐
│              OpenCode 核心 (AI Agent)                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │  AI推理引擎                                         │  │
│  │  - LLM API调用                                     │  │
│  │  - Agent编排                                        │  │
│  │  - Tools执行                                       │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 数据流向（v0.2简化）

```
用户输入 (手机端)
    ↓
React Native UI (HTTP Client)
    ↓
HTTP POST /session/{id}/message
    ↓
OpenCode HTTP Server (REST API)
    ↓
AI Agent推理
    ↓
HTTP Response (消息列表)
    ↓
React Native UI显示
```

---

### 架构对比

| 维度 | v0.1 WebSocket方案 | v0.2 REST API方案 |
|------|------------------|-----------------|
| **复杂度** | 高（WebSocket+子进程管理） | 低（标准REST API） |
| **稳定性** | 中（需要手动管理连接） | 高（OpenCode内置） |
| **调试难度** | 高（双向异步流） | 低（标准HTTP） |
| **扩展性** | 中（需要自定义协议） | 高（OpenAPI规范） |
| **安全性** | 中（需要自定义认证） | 高（Basic Auth + HTTPS） |
| **跨平台** | 中（需要WebSocket库） | 高（标准HTTP库） |

**选择**：✅ v0.2 REST API方案（更稳定、更简单）

---

## 🛠️ 技术实现

### 技术栈（v0.2更新）

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **移动端** | React Native 0.73+ | 跨平台移动应用框架 |
| **移动端通信** | HTTP REST API | 标准RESTful API调用 |
| **OpenCode Server** | opencode serve | OpenCode内置HTTP服务器 |
| **OpenCode API** | OpenAPI 3.1 | 标准RESTful API规范 |
| **HTTP客户端** | axios (React Native) | HTTP请求库 |
| **状态管理** | React Context/Redux | 组件状态管理 |

---

## 📝 详细设计（v0.2简化版）

### 一、OpenCode HTTP Server

#### 1.1 启动OpenCode Server

```bash
# 启动OpenCode HTTP服务器（后台运行）
opencode serve --port 4096 --hostname 0.0.0.0 --cors http://localhost:5173 &

# 可选：设置密码保护
OPENCODE_SERVER_PASSWORD=your-password opencode serve --port 4096
```

#### 1.2 核心API端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/global/health` | 健康检查 |
| `POST` | `/session` | 创建新会话 |
| `GET` | `/session/:id` | 获取会话详情 |
| `GET` | `/session/:id/message` | 获取消息列表 |
| `POST` | `/session/:id/message` | 发送消息 |
| `GET` | `/session/:id/message/:messageId` | 获取单个消息 |
| `DELETE` | `/session/:id` | 删除会话 |
| `GET` | `/project` | 获取项目信息 |
| `GET` | `/file/content` | 读取文件 |

#### 1.3 API调用示例

```bash
# 1. 健康检查
curl http://localhost:4096/global/health
# 返回: {"healthy":true,"version":"1.1.53"}

# 2. 创建会话
curl -X POST http://localhost:4096/session \
  -H "Content-Type: application/json" \
  -d '{"title":"学习Python"}'

# 3. 发送消息
curl -X POST http://localhost:4096/session/{session_id}/message \
  -H "Content-Type: application/json" \
  -d '{
    "parts": [
      {"type": "text", "text": "今天学什么？"}
    ]
  }'

# 4. 获取消息列表
curl http://localhost:4096/session/{session_id}/message
```

---

### 二、移动端架构

#### 2.1 项目结构

```
mobile/
├── src/
│   ├── screens/
│   │   ├── ChatScreen.tsx         # 聊天界面
│   │   └── HomeScreen.tsx          # 主页
│   ├── services/
│   │   └── opencode.ts            # OpenCode API服务
│   ├── components/
│   │   ├── MessageBubble.tsx       # 消息气泡
│   │   └── ChatInput.tsx          # 输入框
│   ├── navigation/
│   │   └── AppNavigator.tsx         # 导航配置
│   ├── App.tsx                    # 应用入口
│   └── index.ts
├── package.json
├── tsconfig.json
├── app.json
└── README.md
```

#### 2.2 OpenCode API服务

```typescript
// src/services/opencode.ts
import axios, { AxiosInstance } from 'axios';

const BASE_URL = 'http://localhost:4096';  // 或服务器的公网IP

export interface MessagePart {
  type: 'text' | 'step-start' | 'reasoning' | 'step-finish';
  text?: string;
}

export interface Message {
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

export interface Session {
  id: string;
  slug: string;
  version: string;
  projectID: string;
  directory: string;
  title: string;
  time: {
    created: number;
    updated: number;
  };
}

export class OpenCodeService {
  private axiosInstance: AxiosInstance;
  private sessionId: string | null = null;

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async healthCheck(): Promise<{ healthy: boolean; version: string }> {
    const response = await this.axiosInstance.get('/global/health');
    return response.data;
  }

  async createSession(title: string = 'New Chat'): Promise<Session> {
    const response = await this.axiosInstance.post('/session', {
      title,
    });
    this.sessionId = response.data.id;
    return response.data;
  }

  async getMessages(sessionId?: string): Promise<{
    info: Message[];
    parts: MessagePart[];
  }> {
    const id = sessionId || this.sessionId;
    if (!id) {
      throw new Error('No session ID');
    }
    
    const response = await this.axiosInstance.get(`/session/${id}/message`);
    return response.data;
  }

  async sendMessage(
    content: string,
    sessionId?: string
  ): Promise<{
    info: Message;
    parts: MessagePart[];
  }> {
    const id = sessionId || this.sessionId;
    if (!id) {
      throw new Error('No session ID');
    }
    
    const response = await this.axiosInstance.post(`/session/${id}/message`, {
      parts: [
        {
          type: 'text',
          text: content,
        },
      ],
    });
    
    return response.data;
  }

  async deleteSession(sessionId?: string): Promise<boolean> {
    const id = sessionId || this.sessionId;
    if (!id) {
      throw new Error('No session ID');
    }
    
    const response = await this.axiosInstance.delete(`/session/${id}`);
    return response.data;
  }
}

export default new OpenCodeService();
```

#### 2.3 聊天界面

```typescript
// src/screens/ChatScreen.tsx
import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';
import OpenCodeService, { Message, MessagePart } from '../services/opencode';

export default function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  
  const scrollViewRef = useRef<ScrollView>(null);
  const opencodeService = OpenCodeService;

  useEffect(() => {
    // 初始化：创建session
    initSession();
  }, []);

  useEffect(() => {
    // 自动滚动到底部
    if (messages.length > 0) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  const initSession = async () => {
    try {
      setLoading(true);
      
      // 创建session
      const session = await opencodeService.createSession('学习Python');
      setSessionId(session.id);
      
      // 健康检查
      const health = await opencodeService.healthCheck();
      setConnected(health.healthy);
      
      setLoading(false);
    } catch (error) {
      console.error('Failed to init session:', error);
      setLoading(false);
    }
  };

  const loadMessages = async () => {
    if (!sessionId) return;
    
    try {
      const data = await opencodeService.getMessages(sessionId);
      
      // 转换消息格式
      const msgs = data.info.map((msg) => ({
        ...msg,
        content: msg.parts[0]?.text || '',
      }));
      
      setMessages(msgs);
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || !sessionId) {
      return;
    }

    try {
      setLoading(true);
      setInput('');
      
      // 发送消息
      const response = await opencodeService.sendMessage(input, sessionId);
      
      // 添加用户消息
      const userMsg: Message = {
        ...response.info,
        role: 'user',
        content: input,
      };
      
      // 添加AI回复（等待完整的parts）
      if (response.parts.length > 0) {
        const aiMsg: Message = {
          ...response.info,
          role: 'assistant',
          content: response.parts[0]?.text || '',
        };
        
        setMessages(prev => [...prev, userMsg, aiMsg]);
      }
      
      setLoading(false);
      Keyboard.dismiss();
    } catch (error) {
      console.error('Failed to send message:', error);
      setLoading(false);
    }
  };

  const renderMessage = (message: Message) => {
    const isUser = message.role === 'user';
    
    return (
      <View
        key={message.info.id}
        style={[
          styles.messageBubble,
          isUser ? styles.userMessage : styles.assistantMessage,
        ]}
      >
        <Text style={[
          styles.messageText,
          isUser ? styles.userText : styles.assistantText,
        ]}>
          {message.content}
        </Text>
        <Text style={styles.messageTime}>
          {new Date(message.info.time.created).toLocaleTimeString()}
        </Text>
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>连接中...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* 头部 */}
        <View style={styles.header}>
          <Text style={styles.title}>OpenCode Mobile</Text>
          <View style={[
            styles.status,
            connected ? styles.connected : styles.disconnected,
          ]}>
            <Text style={styles.statusText}>
              {connected ? '已连接' : '未连接'}
            </Text>
          </View>
        </View>

        {/* 消息列表 */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.messages}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={true}
        >
          {messages.map(renderMessage)}
        </ScrollView>

        {/* 输入框 */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="输入消息..."
            placeholderTextColor="#888"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="send"
            onSubmitEditing={handleSend}
            editable={connected && !loading}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!connected || loading) && styles.disabledButton]}
            onPress={handleSend}
            disabled={!connected || loading}
          >
            {loading ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={styles.sendButtonText}>发送</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
    fontSize: 14,
  },
  keyboardContainer: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 12,
    backgroundColor: '#007AFF',
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  status: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
    backgroundColor: '#555',
  },
  connected: {
    backgroundColor: '#4CAF50',
  },
  disconnected: {
    backgroundColor: '#f44336',
  },
  statusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: 'bold',
  },
  messages: {
    flex: 1,
  },
  messagesContent: {
    paddingVertical: 10,
  },
  messageBubble: {
    maxWidth: '80%',
    marginVertical: 4,
    marginHorizontal: 10,
    padding: 12,
    borderRadius: 16,
  },
  userMessage: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
  },
  assistantMessage: {
    backgroundColor: '#fff',
    alignSelf: 'flex-start',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  messageText: {
    fontSize: 15,
    lineHeight: 20,
  },
  userText: {
    color: '#fff',
  },
  assistantText: {
    color: '#333',
  },
  messageTime: {
    fontSize: 11,
    marginTop: 4,
  },
  inputContainer: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  input: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 8,
    fontSize: 15,
    marginHorizontal: 8,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
```
backend/
├── main.py                 # FastAPI主应用
├── cli_manager.py          # CLI会话管理器
├── requirements.txt        # 依赖列表
└── README.md              # 后端说明文档
```

#### 1.2 依赖文件

```python
# requirements.txt
fastapi==0.110.0
uvicorn==0.27.0
python-multipart==0.0.9
websockets==12.0
pydantic==2.6.0
```

#### 1.3 CLI会话管理器

```python
# cli_manager.py
import subprocess
import threading
import queue
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
import uuid

@dataclass
class CLIMessage:
    """CLI消息"""
    type: str  # "output", "error", "status"
    content: str

class CLISession:
    """CLI会话管理器"""
    
    def __init__(self, session_id: str, workdir: str):
        self.session_id = session_id
        self.workdir = workdir
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.is_running = False
        self.websocket = None
        self._read_thread = None
        self._push_thread = None
    
    async def start_cli(self, command: list) -> Dict:
        """
        启动CLI进程
        
        Args:
            command: CLI命令列表，如 ["claude", "code"]
        
        Returns:
            {"status": "started" | "error", "message": "..."}
        """
        try:
            # 启动子进程
            self.process = subprocess.Popen(
                command,
                cwd=self.workdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                bufsize=1,  # 行缓冲
                universal_newlines=True,  # 文本模式
                shell=False
            )
            
            self.is_running = True
            
            # 启动输出读取线程（在后台线程中运行）
            self._read_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self._read_thread.start()
            
            # 启动WebSocket推送线程（在后台协程中运行）
            self._push_thread = threading.Thread(
                target=asyncio.run,
                args=(self._push_to_websocket(),),
                daemon=True
            )
            self._push_thread.start()
            
            return {
                "status": "started",
                "session_id": self.session_id,
                "command": " ".join(command)
            }
            
        except Exception as e:
            self.is_running = False
            return {
                "status": "error",
                "message": f"Failed to start CLI: {str(e)}"
            }
    
    def _read_output(self):
        """
        读取CLI输出（在线程中运行）
        
        这是一个阻塞操作，所以必须在独立线程中运行
        """
        if not self.process:
            return
        
        try:
            while self.is_running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    # 将输出放入队列
                    self.output_queue.put(CLIMessage(
                        type="output",
                        content=line
                    ))
                else:
                    # EOF，停止读取
                    break
        except Exception as e:
            self.output_queue.put(CLIMessage(
                type="error",
                content=f"Output reading error: {str(e)}"
            ))
        finally:
            # 进程结束，发送状态消息
            if self.process and self.process.poll() is not None:
                self.output_queue.put(CLIMessage(
                    type="status",
                    content="Process terminated"
                ))
    
    async def _push_to_websocket(self):
        """
        推送输出到WebSocket（在协程中运行）
        
        从队列中读取消息，通过WebSocket发送
        """
        if not self.websocket:
            return
        
        try:
            while self.is_running:
                try:
                    # 从队列中获取消息（带超时）
                    message = self.output_queue.get(timeout=0.1)
                    
                    # 通过WebSocket发送
                    if self.websocket:
                        await self.websocket.send_json({
                            "type": message.type,
                            "content": message.content
                        })
                except queue.Empty:
                    # 队列为空，继续等待
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            self.output_queue.put(CLIMessage(
                type="error",
                content=f"WebSocket error: {str(e)}"
            ))
    
    async def send_input(self, input_text: str) -> Dict:
        """
        发送输入到CLI stdin
        
        Args:
            input_text: 用户输入的文本
        
        Returns:
            {"status": "sent" | "error", "message": "..."}
        """
        if not self.process or not self.process.stdin:
            return {
                "status": "error",
                "message": "Process not running"
            }
        
        try:
            # 写入stdin
            self.process.stdin.write(input_text + "\n")
            self.process.stdin.flush()
            
            return {
                "status": "sent",
                "content": input_text
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to send input: {str(e)}"
            }
    
    async def stop_cli(self) -> Dict:
        """
        停止CLI进程
        
        Returns:
            {"status": "stopped" | "error", "message": "..."}
        """
        self.is_running = False
        
        if not self.process:
            return {
                "status": "error",
                "message": "Process not running"
            }
        
        try:
            # 关闭stdin
            if self.process.stdin:
                self.process.stdin.close()
            
            # 终止进程
            self.process.terminate()
            
            # 等待进程结束（最多5秒）
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 强制杀死
                self.process.kill()
                self.process.wait()
            
            return {
                "status": "stopped",
                "session_id": self.session_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to stop CLI: {str(e)}"
            }


class CLISessionManager:
    """CLI会话管理器（单例）"""
    
    _instance: Optional['CLISessionManager'] = None
    _sessions: Dict[str, CLISession] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def create_session(cls, session_id: str, workdir: str) -> CLISession:
        """创建新会话"""
        session = CLISession(session_id, workdir)
        cls._sessions[session_id] = session
        return session
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[CLISession]:
        """获取会话"""
        return cls._sessions.get(session_id)
    
    @classmethod
    def remove_session(cls, session_id: str) -> bool:
        """移除会话"""
        if session_id in cls._sessions:
            del cls._sessions[session_id]
            return True
        return False
    
    @classmethod
    def list_sessions(cls) -> Dict:
        """列出所有会话"""
        return {
            "sessions": list(cls._sessions.keys()),
            "count": len(cls._sessions)
        }
```

#### 1.4 FastAPI主应用

```python
# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional
import uuid

from cli_manager import CLISessionManager, CLISession

# 创建FastAPI应用
app = FastAPI(
    title="Skills System Mobile Backend",
    description="WebSocket代理服务，允许移动端通过WebSocket访问Claude Code CLI",
    version="0.1.0"
)

# CORS配置（允许React Native访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取会话管理器单例
session_manager = CLISessionManager()

# ======== WebSocket端点 ========

@app.websocket("/ws/terminal/{session_id}")
async def websocket_terminal(websocket: WebSocket, session_id: str):
    """
    WebSocket终端端点
    
    客户端通过此端点连接到终端会话
    支持双向通信：
    - 客户端 → 服务器：{"type": "start" | "input" | "stop", ...}
    - 服务器 → 客户端：{"type": "output" | "error" | "status", ...}
    """
    await websocket.accept()
    
    # 创建或获取会话
    session = session_manager.get_session(session_id)
    if session is None:
        # 创建新会话
        session = session_manager.create_session(
            session_id=session_id,
            workdir="D:/AI/2026/LearningSystem"  # Skills系统目录
        )
    
    # 关联WebSocket
    session.websocket = websocket
    
    try:
        # 主循环：接收客户端消息并处理
        while True:
            # 接收消息
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "start":
                # 启动CLI
                command = message.get("command", ["claude", "code"])
                result = await session.start_cli(command)
                
                # 发送响应
                await websocket.send_json({
                    "type": "status",
                    "content": result
                })
                
            elif message_type == "input":
                # 发送输入到CLI
                input_text = message.get("content", "")
                result = await session.send_input(input_text)
                
                # 发送响应
                await websocket.send_json({
                    "type": "status",
                    "content": result
                })
                
            elif message_type == "stop":
                # 停止CLI
                result = await session.stop_cli()
                
                # 发送响应
                await websocket.send_json({
                    "type": "status",
                    "content": result
                })
                
                # 移除会话
                session_manager.remove_session(session_id)
                
                # 关闭WebSocket
                break
                
            else:
                # 未知消息类型
                await websocket.send_json({
                    "type": "error",
                    "content": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        # 发送错误消息
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        # 清理会话
        if session_manager.get_session(session_id):
            await session.stop_cli()
            session_manager.remove_session(session_id)
        print(f"Session cleaned up: {session_id}")


# ======== 管理API ========

@app.get("/api/sessions")
async def list_sessions():
    """
    列出所有活跃会话
    
    GET /api/sessions
    Response: {"sessions": ["session_001", ...], "count": 2}
    """
    return session_manager.list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session_status(session_id: str):
    """
    获取指定会话的状态
    
    GET /api/sessions/{session_id}
    Response: {
        "session_id": "session_001",
        "is_running": true,
        "workdir": "D:/AI/2026/LearningSystem"
    }
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "is_running": session.is_running,
        "workdir": session.workdir
    }


@app.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str):
    """
    停止指定会话
    
    DELETE /api/sessions/{session_id}
    Response: {"status": "stopped", "session_id": "session_001"}
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = await session.stop_cli()
    session_manager.remove_session(session_id)
    
    return result


# ======== 健康检查 ========

@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    GET /health
    Response: {"status": "healthy", "sessions": 2}
    """
    return {
        "status": "healthy",
        "sessions": len(session_manager._sessions)
    }


if __name__ == "__main__":
    # 开发环境运行
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

#### 1.5 启动脚本

**Linux/Mac**:
```bash
#!/bin/bash
# start.sh

echo "Starting Skills System Mobile Backend..."

# 激活虚拟环境（如果使用）
# source venv/bin/activate

# 启动FastAPI服务器
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Windows**:
```batch
@echo off
REM start.bat

echo Starting Skills System Mobile Backend...

REM 激活虚拟环境（如果使用）
REM venv\Scripts\activate

REM 启动FastAPI服务器
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### 二、移动端设计

#### 2.1 项目结构

```
mobile/
├── src/
│   ├── components/
│   │   ├── TerminalOutput.tsx    # 终端输出组件
│   │   ├── TerminalInput.tsx     # 终端输入组件
│   │   └── ConnectionStatus.tsx  # 连接状态组件
│   ├── screens/
│   │   ├── TerminalScreen.tsx    # 终端主界面
│   │   └── HomeScreen.tsx        # 主页
│   ├── services/
│   │   └── websocket.ts          # WebSocket服务
│   ├── navigation/
│   │   └── AppNavigator.tsx      # 导航配置
│   ├── App.tsx                   # 应用入口
│   └── index.ts                  # 入口文件
├── package.json
├── tsconfig.json
├── app.json
└── README.md
```

#### 2.2 依赖文件

```json
{
  "dependencies": {
    "react": "18.2.0",
    "react-native": "0.73.0",
    "websocket": "^1.0.34",
    "@react-navigation/native": "^6.1.9",
    "@react-navigation/native-stack": "^6.9.17"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/websocket": "^1.0.10",
    "typescript": "^5.0.0"
  }
}
```

#### 2.3 WebSocket服务

```typescript
// src/services/websocket.ts
import { w3cwebsocket as W3CWebSocket } from 'websocket';

export type MessageType = 
  | 'start'     // 启动CLI
  | 'input'     // 发送输入
  | 'stop'      // 停止CLI
  | 'output'    // CLI输出
  | 'error'     // 错误消息
  | 'status';   // 状态消息

export interface WSMessage {
  type: MessageType;
  content?: string;
  command?: string[];
  cwd?: string;
}

export interface WSConfig {
  url: string;
  sessionId: string;
  command: string[];
  cwd: string;
}

export class WebSocketService {
  private ws: W3CWebSocket | null = null;
  private config: WSConfig;
  private messageHandlers: Map<MessageType, (msg: WSMessage) => void> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;

  constructor(config: WSConfig) {
    this.config = config;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const url = `${this.config.url}/ws/terminal/${this.config.sessionId}`;
        this.ws = new W3CWebSocket(url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          
          // 自动发送启动命令
          this.sendMessage({
            type: 'start',
            command: this.config.command,
            cwd: this.config.cwd
          });
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WSMessage = JSON.parse(event.data.toString());
            console.log('Received:', message);
            
            // 调用对应的消息处理器
            const handler = this.messageHandlers.get(message.type);
            if (handler) {
              handler(message);
            }
          } catch (error) {
            console.error('Failed to parse message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket closed');
          
          // 尝试重连
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
              console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
              this.connect().catch(console.error);
            }, this.reconnectDelay);
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  onMessage(type: MessageType, handler: (msg: WSMessage) => void) {
    this.messageHandlers.set(type, handler);
  }

  sendMessage(message: WSMessage) {
    if (this.ws && this.ws.readyState === W3CWebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  }

  sendInput(input: string) {
    this.sendMessage({
      type: 'input',
      content: input
    });
  }

  stop() {
    this.sendMessage({ type: 'stop' });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === W3CWebSocket.OPEN;
  }
}
```

#### 2.4 终端主界面

```typescript
// src/screens/TerminalScreen.tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView
} from 'react-native';
import { WebSocketService, WSMessage } from '../services/websocket';

const WS_URL = 'ws://localhost:8000';

export default function TerminalScreen() {
  const [output, setOutput] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  
  const scrollViewRef = useRef<ScrollView>(null);
  const wsServiceRef = useRef<WebSocketService | null>(null);

  useEffect(() => {
    // 创建WebSocket服务
    wsServiceRef.current = new WebSocketService({
      url: WS_URL,
      sessionId: sessionId,
      command: ['claude', 'code'],
      cwd: 'D:/AI/2026/LearningSystem'
    });

    // 设置消息处理器
    wsServiceRef.current.onMessage('output', (msg: WSMessage) => {
      if (msg.content) {
        setOutput(prev => [...prev, msg.content!]);
        // 自动滚动到底部
        setTimeout(() => {
          scrollViewRef.current?.scrollToEnd({ animated: true });
        }, 100);
      }
    });

    wsServiceRef.current.onMessage('error', (msg: WSMessage) => {
      if (msg.content) {
        setOutput(prev => [...prev, `ERROR: ${msg.content}`]);
      }
    });

    wsServiceRef.current.onMessage('status', (msg: WSMessage) => {
      console.log('Status:', msg.content);
      if (msg.content?.status === 'started') {
        setConnected(true);
      } else if (msg.content?.status === 'stopped') {
        setConnected(false);
      }
    });

    // 连接WebSocket
    wsServiceRef.current.connect().catch(console.error);

    return () => {
      // 清理
      wsServiceRef.current?.disconnect();
    };
  }, [sessionId]);

  const handleSend = () => {
    if (!input.trim() || !wsServiceRef.current?.isConnected()) {
      return;
    }

    // 显示用户输入
    setOutput(prev => [...prev, `$ ${input}`]);

    // 发送到WebSocket
    wsServiceRef.current.sendInput(input);

    setInput('');
    Keyboard.dismiss();
  };

  const handleStop = () => {
    wsServiceRef.current?.stop();
    setConnected(false);
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* 头部 */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.title}>Claude Code 终端</Text>
            <Text style={styles.subtitle}>Skills v3.0</Text>
          </View>
          <View style={[styles.status, connected ? styles.connected : styles.disconnected]}>
            <Text style={styles.statusText}>
              {connected ? '已连接' : '未连接'}
            </Text>
          </View>
        </View>

        {/* 终端输出 */}
        <View style={styles.outputContainer}>
          <ScrollView
            ref={scrollViewRef}
            style={styles.output}
            contentContainerStyle={styles.outputContent}
            showsVerticalScrollIndicator={true}
          >
            {output.map((line, index) => (
              <Text key={index} style={styles.outputLine}>
                {line}
              </Text>
            ))}
            {output.length === 0 && (
              <Text style={styles.placeholder}>等待连接...</Text>
            )}
          </ScrollView>
        </View>

        {/* 输入区域 */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="输入命令..."
            placeholderTextColor="#888"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="send"
            onSubmitEditing={handleSend}
            editable={connected}
          />
          <TouchableOpacity
            style={[styles.sendButton, !connected && styles.disabledButton]}
            onPress={handleSend}
            disabled={!connected}
          >
            <Text style={styles.sendButtonText}>发送</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.stopButton, connected && styles.activeStopButton]}
            onPress={handleStop}
          >
            <Text style={styles.stopButtonText}>停止</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1e1e1e',
  },
  keyboardContainer: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 12,
    backgroundColor: '#2d2d2d',
    borderBottomWidth: 1,
    borderBottomColor: '#444',
  },
  headerLeft: {
    flex: 1,
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  subtitle: {
    color: '#888',
    fontSize: 12,
    marginTop: 2,
  },
  status: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
  },
  connected: {
    backgroundColor: '#4CAF50',
  },
  disconnected: {
    backgroundColor: '#f44336',
  },
  statusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: 'bold',
  },
  outputContainer: {
    flex: 1,
  },
  output: {
    flex: 1,
    paddingHorizontal: 10,
    paddingVertical: 10,
  },
  outputContent: {
    paddingBottom: 10,
  },
  outputLine: {
    color: '#d4d4d4',
    fontSize: 13,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'Courier New',
    marginBottom: 1,
  },
  placeholder: {
    color: '#666',
    fontSize: 14,
    fontStyle: 'italic',
  },
  inputContainer: {
    flexDirection: 'row',
    paddingHorizontal: 10,
    paddingVertical: 12,
    backgroundColor: '#2d2d2d',
    borderTopWidth: 1,
    borderTopColor: '#444',
    gap: 8,
  },
  input: {
    flex: 1,
    backgroundColor: '#3c3c3c',
    color: '#d4d4d4',
    borderWidth: 1,
    borderColor: '#555',
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'Courier New',
  },
  sendButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    minWidth: 60,
  },
  disabledButton: {
    backgroundColor: '#555',
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  stopButton: {
    backgroundColor: '#555',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    minWidth: 50,
  },
  activeStopButton: {
    backgroundColor: '#FF5722',
  },
  stopButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
```

---

## 📋 实施计划（v0.2更新）

### 阶段1：OpenCode Server验证（1天）✅ 已完成

**目标**：验证OpenCode HTTP Server可行性

- [x] Day 1: 实验验证（已验证成功）
  - ✅ 启动OpenCode HTTP Server
  - ✅ 健康检查API测试
  - ✅ 创建session API测试
  - ✅ 发送message API测试
  - ✅ 获取message列表API测试

**验收标准**：
- ✅ OpenCode HTTP Server可正常启动
- ✅ REST API可正常访问
- ✅ 消息发送和接收功能正常

---

### 阶段2：移动端开发（3-5天）

**目标**：完成React Native应用，支持OpenCode HTTP API

- [ ] Day 1: 项目初始化和依赖安装
  - [ ] 创建React Native项目
  - [ ] 安装axios和导航依赖
  - [ ] 配置TypeScript

- [ ] Day 2: 实现OpenCode API服务
  - [ ] 封装HTTP客户端（axios）
  - [ ] 实现session管理
  - [ ] 实现message发送和接收

- [ ] Day 3: 实现聊天界面UI
  - [ ] MessageBubble组件（消息气泡）
  - [ ] ChatInput组件（输入框）
  - [ ] ChatScreen主界面

- [ ] Day 4: 集成API和UI
  - [ ] 消息列表展示
  - [ ] 发送消息功能
  - [ ] 自动滚动到底部

- [ ] Day 5: 测试和优化
  - [ ] 真机测试
  - [ ] 性能优化
  - [ ] 错误处理

**验收标准**：
- ✅ 能连接到OpenCode HTTP Server
- ✅ 能实时显示对话消息
- ✅ 能发送消息并接收回复
- ✅ 支持消息历史
- ✅ 支持自动滚动

---

### 阶段3：OpenCode Server配置（1-2天）

**目标**：配置OpenCode Server在本地PC/服务器上

- [ ] Windows配置
  - [ ] 启动脚本（bat）
  - [ ] 开机自启动（可选）
  - [ ] 防火墙配置

- [ ] Mac mini配置
  - [ ] 启动脚本（sh）
  - [ ] launchd配置（开机自启动）
  - [ ] 网络配置（固定IP/内网穿透）

**验收标准**：
- ✅ Windows能自动启动OpenCode Server
- ✅ Mac mini能自动启动OpenCode Server
- ✅ 移动端能正常访问

---

### 阶段4：部署和测试（1-2天）

**目标**：真实环境部署和测试

- [ ] 网络配置
  - [ ] 局域网测试
  - [ ] 公网IP配置（可选）
  - [ ] HTTPS配置（可选）

- [ ] 真实场景测试
  - [ ] 长时间运行测试（24小时+）
  - [ ] 多会话测试
  - [ ] 性能测试

**验收标准**：
- ✅ 移动端能稳定连接
- ✅ OpenCode Server稳定运行
- ✅ 性能满足使用要求

---

### 阶段3：集成测试（2-3天）

**目标**：端到端测试，真实场景验证

- [ ] Day 1: 本地网络环境测试
- [ ] Day 2: 长时间运行测试
- [ ] Day 3: 多会话并发测试

**验收标准**：
- ✅ 在同一局域网内能正常使用
- ✅ 能稳定运行1小时以上
- ✅ 支持至少3个并发会话

---

### 阶段4：优化和部署（2-3天）

**目标**：性能优化和正式部署

- [ ] Day 1: 性能优化（减少延迟、优化推送）
- [ ] Day 2: 部署配置（启动脚本、环境变量）
- [ ] Day 3: 打包和发布

**验收标准**：
- ✅ WebSocket延迟 < 100ms
- ✅ 后端能稳定运行24小时
- ✅ 移动端APK可正常安装

---

## ❓ 待确认问题（v0.2更新）

### ✅ 已解决的问题

#### 问题1：CLI相关问题（✅ 已解决）

**答案**：OpenCode HTTP Server完美支持！

**验证结果**：
- ✅ OpenCode支持`opencode serve`命令，启动HTTP Server
- ✅ 提供完整的RESTful API（OpenAPI 3.1规范）
- ✅ 支持session管理、message发送、消息历史查询
- ✅ 完全无需通过subprocess捕获stdout/stderr
- ✅ 输出格式：JSON（结构化消息）

**不需要**：
- ❌ WebSocket + subprocess复杂方案
- ❌ 自建FastAPI后端
- ❌ 捕获stdin/stdout

**新架构**：
```
移动端 → HTTP REST API → OpenCode HTTP Server → AI Agent
```

---

### 🎯 已确定的设计

#### 2. 部署环境（✅ 已确定）

**Windows**：
- 启动命令：`opencode serve --port 4096 --hostname 0.0.0.0`
- 启动脚本：`start_opencode.bat`
- 开机自启动：可选（任务计划程序）

**Mac mini**：
- 启动命令：`opencode serve --port 4096 --hostname 0.0.0.0`
- 启动脚本：`start_opencode.sh`
- 开机自启动：launchd配置

**网络访问**：
- 局域网：移动端和PC/服务器同一WiFi即可
- 公网IP：可选（需要路由器端口转发）

#### 3. 功能需求（✅ 已确定）

**MVP功能（v0.2）**：
- ✅ 创建session
- ✅ 发送消息
- ✅ 接收AI回复
- ✅ 显示消息历史
- ✅ 自动滚动

**v0.3功能（以后实现）**：
- ⏸️ 多会话并发（低优先级）
- ⏸️ 会话持久化（OpenCode本身记录历史）
- ⏸️ 文件传输（OpenCode API支持）
- ⏸️ 彩色输出（OpenCode API支持）
- ⏸️ 终端快捷键（移动端不需要）

#### 4. 用户体验（✅ MVP不强要求）

**MVP体验（v0.2）**：
- ✅ 简洁的聊天界面
- ✅ 消息气泡（用户/AI区分）
- ✅ 输入框 + 发送按钮
- ✅ 自动滚动
- ✅ 连接状态显示

**v0.3优化（以后实现）**：
- ⏸️ 命令历史（OpenCode本身支持）
- ⏸️ 自动补全（OpenCode本身支持）
- ⏸️ 多标签页（以后考虑）
- ⏸️ 字体大小调整（以后考虑）

---

### 📋 下一步行动

#### 立即行动（今天）

- [ ] 更新设计文档为v0.2（✅ 已完成）
- [ ] 提交设计文档到Git
- [ ] 创建移动端项目
- [ ] 实现OpenCode API服务

#### 短期目标（本周）

- [ ] 完成阶段2（移动端开发）
- [ ] Windows本地测试
- [ ] Mac mini部署测试

#### 中期目标（下周）

- [ ] 完成阶段3-4
- [ ] 真实场景测试
- [ ] 性能优化

---

## 🎯 后续迭代方向

### v0.2 功能增强

- [ ] 支持ANSI颜色代码解析和显示
- [ ] 支持终端快捷键（Ctrl+C、Ctrl+D等）
- [ ] 支持命令历史（上下箭头浏览）
- [ ] 支持自动滚动和手动滚动切换
- [ ] 支持清屏命令

### v0.3 性能优化

- [ ] WebSocket消息压缩
- [ ] 输出缓冲和批量推送
- [ ] 移动端虚拟键盘优化
- [ ] 长连接保活机制

### v0.4 用户体验

- [ ] 支持多标签页（多个会话）
- [ ] 支持会话持久化
- [ ] 支持文件传输
- [ ] 支持截图和分享

### v0.5 高级功能

- [ ] 支持语音输入（Speech-to-Text）
- [ ] 支持TTS输出（Text-to-Speech）
- [ ] 支持离线模式（缓存历史记录）
- [ ] 支持代码高亮

---

## 📚 参考资料

- [FastAPI WebSocket文档](https://fastapi.tiangolo.com/advanced/websockets/)
- [Python subprocess文档](https://docs.python.org/3/library/subprocess.html)
- [React Native WebSocket](https://github.com/websockets/ws)
- [WebSocket协议RFC 6455](https://datatracker.ietf.org/doc/html/rfc6455)

---

**文档维护**：returnfortheking  
**最后更新**：2026-02-08  
**下次更新**：待确认问题解答后
