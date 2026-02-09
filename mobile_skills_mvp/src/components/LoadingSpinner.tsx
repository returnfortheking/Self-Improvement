/**
 * Loading Spinner - 加载指示器组件
 */

import React from 'react';
import {
  View,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';

export interface LoadingSpinnerProps {
  size?: 'small' | 'large';
  color?: string;
  text?: string;
}

export default function LoadingSpinner({
  size = 'large',
  color = '#007AFF',
  text,
}: LoadingSpinnerProps): JSX.Element {
  return (
    <View style={styles.container}>
      <ActivityIndicator
        size={size === 'small' ? 'small' : 'large'}
        color={color || '#007AFF'}
        style={[
          styles.indicator,
          text && styles.textContainer,
        ]}
      >
        {text && (
          <Text style={styles.text}>{text}</Text>
        )}
      </ActivityIndicator>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 10,
  },
  indicator: {
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
  },
  textContainer: {
    marginTop: 10,
    marginLeft: 8,
  },
  text: {
    fontSize: 12,
    color: '#666',
  },
});
