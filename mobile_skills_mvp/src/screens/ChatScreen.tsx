/**
 * Chat Screen - OpenCode聊天界面（完整功能版）
 */

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
  Alert,
  Modal,
} from 'react-native';
import { OpenCodeServiceSimple } from '../services/openCodeSimple';
import type { SimpleMessage } from '../services/openCodeSimple';
import { ServerManager } from '../services/ServerManager';
import { DEFAULT_SERVERS } from '../services/servers.config';

export default function ChatScreen() {
  const [messages, setMessages] = useState<SimpleMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState('https://rousingly-childlike-latarsha.ngrok-free.dev'); // ngrok公网地址
  const [currentServer, setCurrentServer] = useState<string>('');
  const [serverManager] = useState(() => new ServerManager(DEFAULT_SERVERS));

  const scrollViewRef = useRef<ScrollView>(null);
  const openCodeService = useRef<OpenCodeServiceSimple | null>(null);

  console.log('[ChatScreen] Render - State:', {
    connected,
    connecting,
    loading,
    sending,
    sessionId,
    messagesCount: messages.length,
  });

  // 初始化连接
  useEffect(() => {
    initConnection();
  }, []);

  // 初始化：创建session
  const initConnection = async () => {
    try {
      setConnecting(true);
      setError(null);

      console.log('[ChatScreen] Initializing connection with ServerManager...');

      // 使用 ServerManager 选择最佳服务器
      const bestServer = await serverManager.selectBestServer();

      if (!bestServer) {
        throw new Error('没有可用的服务器。请检查网络连接或服务器配置。');
      }

      console.log('[ChatScreen] Selected server:', bestServer.name, bestServer.url);

      // 更新服务实例的 BASE_URL
      openCodeService.current = new OpenCodeServiceSimple(bestServer.url);
      setServerUrl(bestServer.url);
      setCurrentServer(bestServer.name);

      console.log('[ChatScreen] Creating session...');

      const session = await openCodeService.current.createSession('学习Python');

      console.log('[ChatScreen] Session created:', session.id);

      setSessionId(session.id);

      // 等待1秒后加载消息（不需要保留本地消息，因为是首次加载）
      await new Promise(resolve => setTimeout(resolve, 1000));

      await loadMessages(session.id, false);
      setConnected(true);
      setConnecting(false);
      setLoading(false);

      console.log('[ChatScreen] Connection successful!');
    } catch (err: any) {
      console.error('[ChatScreen] Failed to init session:', err);
      setError(`无法连接到OpenCode Server：${err.message || '未知错误'}`);
      setConnecting(false);
      setLoading(false);
      setConnected(false);
      setSessionId(null);
      Alert.alert(
        '连接失败',
        `无法连接到OpenCode Server\n\n错误：${err.message || '未知错误'}\n\n可能原因：\n1. Mac mini 服务器未启动\n2. ngrok 隧道未建立\n3. 网络连接问题`,
        [
          { text: '取消', style: 'cancel' },
          { text: '设置', onPress: () => setShowSettings(true) },
          { text: '重试', onPress: () => initConnection() },
        ]
      );
    }
  };

  // 加载消息
  const loadMessages = async (id: string, preserveLocalMessages: boolean = false) => {
    if (!id || !openCodeService.current) return;

    try {
      setLoading(true);
      const data = await openCodeService.current.getMessages(id);

      const serverMsgs = data.info
        .filter((msg: any) => msg.id) // 过滤掉没有 id 的消息
        .map((msg: any) => ({
          id: msg.id,
          role: msg.role || 'user', // 默认为 user
          content: msg.content || '',
          time: msg.time || Date.now(),
        }));

      console.log('[ChatScreen] Server messages count:', serverMsgs.length);
      console.log('[ChatScreen] Server messages:', serverMsgs);

      if (preserveLocalMessages) {
        // 智能合并：保留本地用户消息，合并服务器消息
        setMessages(prev => {
          // 保留本地临时消息（id以user_开头且不在服务器列表中）
          const localMessages = prev.filter(msg =>
            msg.id && msg.id.startsWith('user_') && !serverMsgs.some((sm: any) => sm.time === msg.time)
          );

          console.log('[ChatScreen] Preserved local messages:', localMessages.length);

          // 合并：本地消息 + 服务器消息
          const merged = [...localMessages, ...serverMsgs];

          console.log('[ChatScreen] Merged messages count:', merged.length);
          return merged;
        });
      } else {
        // 直接使用服务器消息
        setMessages(serverMsgs);
      }

      setLoading(false);
    } catch (err: any) {
      console.error('[ChatScreen] Failed to load messages:', err);
      setError(`加载消息失败：${err.message || '未知错误'}`);
      setLoading(false);
    }
  };

  // 发送消息
  const handleSend = async () => {
    if (!input.trim() || !connected || loading || sending || !sessionId || !openCodeService.current) {
      if (!connected) {
        setError('请先连接到OpenCode Server');
      }
      return;
    }

    try {
      setSending(true);
      setError(null);
      const messageContent = input;
      setInput('');

      // 添加用户消息到UI（立即显示）
      const userMsg: SimpleMessage = {
        id: `user_${Date.now()}`,
        role: 'user',
        content: messageContent,
        time: Date.now(),
      };

      setMessages(prev => [...prev, userMsg]);

      console.log('[ChatScreen] Sending message:', messageContent);

      // 发送到OpenCode
      const response = await openCodeService.current.sendMessage(sessionId, messageContent);

      console.log('[ChatScreen] Message sent, response:', response);
      console.log('[ChatScreen] Response data:', JSON.stringify(response, null, 2));

      // 重新加载消息以获取AI回复（保留本地消息）
      await new Promise(resolve => setTimeout(resolve, 1500));
      await loadMessages(sessionId, true);

      setSending(false);
      Keyboard.dismiss();

      // 滚动到底部
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    } catch (err: any) {
      console.error('[ChatScreen] Failed to send message:', err);
      setError(`发送失败：${err.message || '未知错误'}`);
      setSending(false);
    }
  };

  // 重新连接
  const handleReconnect = () => {
    initConnection();
  };

  // 更新服务器地址
  const handleUpdateServer = () => {
    setShowSettings(false);
    initConnection();
  };

  // 渲染消息
  const renderMessage = (msg: SimpleMessage, index: number) => {
    const isUser = msg.role === 'user';

    return (
      <View
        key={msg.id}
        style={[
          styles.messageBubble,
          isUser ? styles.userMessage : styles.assistantMessage,
        ]}
      >
        <Text
          style={[
            styles.messageText,
            isUser ? styles.userText : styles.assistantText,
          ]}
        >
          {msg.content}
        </Text>
      </View>
    );
  };

  // 渲染设置弹窗
  const renderSettings = () => (
    <Modal
      visible={showSettings}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowSettings(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalContent}>
          <Text style={styles.modalTitle}>服务器设置</Text>

          <Text style={styles.label}>服务器地址：</Text>
          <TextInput
            style={styles.input}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="http://192.168.x.x:4096"
            placeholderTextColor="#888"
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={styles.hint}>
            当前使用局域网IP访问。请确保：{'\n'}
            1. 手机和PC在同一WiFi{'\n'}
            2. OpenCode Server已启动{'\n'}
            3. IP地址正确
          </Text>

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.button, styles.cancelButton]}
              onPress={() => setShowSettings(false)}
            >
              <Text style={styles.buttonText}>取消</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, styles.confirmButton]}
              onPress={handleUpdateServer}
            >
              <Text style={styles.buttonText}>保存并重连</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  // 连接中界面
  if (connecting) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.center}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>正在连接到 OpenCode Server...</Text>
          <Text style={styles.subtext}>服务器：{serverUrl}</Text>
        </View>
      </SafeAreaView>
    );
  }

  // 主界面
  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* 顶部栏 */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.title}>OpenCode Mobile</Text>
            <View style={[
              styles.statusBadge,
              connected ? styles.connected : styles.disconnected
            ]}>
              <Text style={styles.statusText}>
                {connected ? '已连接' : '未连接'}
              </Text>
              {connected && currentServer && (
                <Text style={styles.serverName}> ({currentServer})</Text>
              )}
            </View>
          </View>
          <TouchableOpacity
            style={styles.settingsButton}
            onPress={() => setShowSettings(true)}
          >
            <Text style={styles.settingsIcon}>⚙️</Text>
          </TouchableOpacity>
        </View>

        {/* 错误提示 */}
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorTitle}>连接错误</Text>
            <Text style={styles.errorMessage}>{error}</Text>
            <View style={styles.errorButtons}>
              <TouchableOpacity
                style={styles.errorButtonSmall}
                onPress={() => setError(null)}
              >
                <Text style={styles.errorButtonText}>关闭</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.reconnectButton}
                onPress={handleReconnect}
              >
                <Text style={styles.reconnectButtonText}>重新连接</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* 消息列表 */}
        <View style={styles.messagesContainer}>
          <ScrollView
            ref={scrollViewRef}
            style={styles.messages}
            contentContainerStyle={styles.messagesContent}
            showsVerticalScrollIndicator={true}
          >
            {messages.map(renderMessage)}
            {messages.length === 0 && !loading && (
              <Text style={styles.placeholder}>
                {connected
                  ? '发送消息开始对话...'
                  : '等待连接 OpenCode Server...'}
              </Text>
            )}
            {loading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="small" color="#007AFF" />
                <Text style={styles.loadingText}>加载中...</Text>
              </View>
            )}
          </ScrollView>
        </View>

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
            editable={connected && !loading && !sending}
            multiline={true}
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!connected || loading || sending) && styles.disabledButton,
            ]}
            onPress={handleSend}
            disabled={!connected || loading || sending}
          >
            {sending ? (
              <ActivityIndicator size="small" color="#fff" />
            ) : (
              <Text style={styles.sendButtonText}>发送</Text>
            )}
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>

      {/* 设置弹窗 */}
      {renderSettings()}
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
    padding: 20,
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
    fontSize: 14,
  },
  subtext: {
    marginTop: 5,
    color: '#999',
    fontSize: 12,
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
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginRight: 10,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
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
  serverName: {
    color: '#fff',
    fontSize: 10,
    marginLeft: 4,
  },
  settingsButton: {
    padding: 5,
  },
  settingsIcon: {
    fontSize: 20,
    color: '#fff',
  },
  errorContainer: {
    padding: 15,
    backgroundColor: 'rgba(255, 0, 0, 0.9)',
    margin: 15,
    borderRadius: 8,
  },
  errorTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  errorMessage: {
    color: '#fff',
    fontSize: 14,
    marginBottom: 12,
  },
  errorButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  errorButtonSmall: {
    backgroundColor: '#d9534f',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 6,
    marginRight: 10,
  },
  errorButtonText: {
    color: '#fff',
    fontSize: 14,
  },
  reconnectButton: {
    backgroundColor: '#FF5722',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 6,
  },
  reconnectButtonText: {
    color: '#fff',
    fontSize: 14,
  },
  messagesContainer: {
    flex: 1,
  },
  messages: {
    flex: 1,
  },
  messagesContent: {
    paddingVertical: 10,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginVertical: 4,
    marginHorizontal: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  userMessage: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
  },
  assistantMessage: {
    backgroundColor: '#fff',
    alignSelf: 'flex-start',
  },
  messageText: {
    fontSize: 15,
    lineHeight: 20,
    color: '#333',
  },
  userText: {
    color: '#fff',
  },
  assistantText: {
    color: '#333',
  },
  placeholder: {
    color: '#999',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 50,
    fontStyle: 'italic',
  },
  loadingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
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
    marginRight: 10,
    maxHeight: 100,
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
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    width: '85%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#333',
  },
  label: {
    fontSize: 16,
    color: '#333',
    marginBottom: 8,
    fontWeight: '500',
  },
  hint: {
    fontSize: 13,
    color: '#666',
    marginTop: 12,
    marginBottom: 20,
    lineHeight: 20,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  button: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#999',
    marginRight: 10,
  },
  confirmButton: {
    backgroundColor: '#007AFF',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
