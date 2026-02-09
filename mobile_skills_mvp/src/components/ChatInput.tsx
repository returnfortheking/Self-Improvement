/**
 * Chat Input - 聊天输入框组件
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Keyboard,
} from 'react-native';

export interface ChatInputProps {
  onSend: (text: string) => void;
  onClear: () => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  multiline?: boolean;
}

export default function ChatInput({
  onSend,
  onClear,
  disabled = false,
  placeholder = "输入消息...",
  maxLength = 500,
  multiline = true,
}: ChatInputProps): JSX.Element {
  const [text, setText] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<TextInput>(null);

  const handleFocus = () => {
    setIsFocused(true);
    inputRef.current?.focus();
  };

  const handleBlur = () => {
    setIsFocused(false);
  };

  const handleSend = () => {
    if (!text.trim() || disabled) {
      return;
    }

    onSend(text);
    setText('');
    Keyboard.dismiss();
  };

  const handleClear = () => {
    if (!text) {
      return;
    }

    onClear();
    setText('');
    if (inputRef.current) {
      inputRef.current?.clear();
    }
  };

  const handleKeyPress = (e: any) => {
    if (e.nativeEvent.key === 'Enter' && !disabled) {
      handleSend();
    }
    };

  return (
    <View style={styles.container}>
      <TextInput
        ref={inputRef}
        style={[
          styles.input,
          disabled && styles.disabledInput,
          isFocused && styles.focusedInput,
        ]}
        value={text}
        onChangeText={setText}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onSubmitEditing={handleSend}
        placeholder={placeholder}
        placeholderTextColor="#888"
        autoCapitalize="none"
        autoCorrect={false}
        maxLength={maxLength}
        multiline={multiline}
        returnKeyType="send"
        enablesReturnKeyAutomatic={true}
      />
      
      {!disabled && (
        <TouchableOpacity
          style={styles.clearButton}
          onPress={handleClear}
          disabled={!text}
        >
          <Text style={styles.clearButtonText}>✕</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
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
  disabledInput: {
    backgroundColor: '#e0e0e0',
    color: '#999',
  },
  focusedInput: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#007AFF',
  },
  clearButton: {
    padding: 10,
    borderRadius: 8,
    backgroundColor: 'rgba(255, 0, 0, 0.8)',
  },
  clearButtonText: {
    color: '#666',
    fontSize: 16,
  },
});
