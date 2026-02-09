/**
 * Message Bubble - 消息气泡组件
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
} from 'react-native';

export interface MessageBubbleProps {
  message: {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    time?: {
      created: number;
      completed?: number;
    };
  };
  onLongPress?: () => void;
}

export default function MessageBubble({ message, isUser = false }: MessageBubbleProps): JSX.Element {
  const { id, role, content, time } = message;
  const showTime = time && isUser && time?.created !== time?.completed;

  const handleLongPress = () => {
    onLongPress && onLongPress();
  };

  return (
    <View
      style={[
        styles.messageBubble,
        isUser ? styles.userMessage : styles.assistantMessage,
      ]}
      onLongPress={handleLongPress}
    >
      {/* 消息时间 */}
      {showTime && (
        <Text style={styles.messageTime}>
          {new Date(time?.created).toLocaleTimeString()}
        </Text>
      )}
      {/* 消息内容 */}
      <Text
        style={[
          styles.messageText,
          isUser ? styles.userText : styles.assistantText,
        ]}
      >
        {content}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  messageBubble: {
    maxWidth: '80%',
    marginVertical: 4,
    marginHorizontal: 10,
    padding: 12,
    borderRadius: 16,
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
    fontSize: 15,
    lineHeight: 20,
  },
  assistantText: {
    color: '#333',
    fontSize: 15,
    lineHeight: 20,
  },
  messageTime: {
    fontSize: 11,
    marginTop: 4,
    color: '#999',
  },
});
