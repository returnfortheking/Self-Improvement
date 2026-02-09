/**
 * Main App - 应用入口
 * 直接渲染ChatScreen（暂时移除导航以简化）
 */

import React from 'react';
import ChatScreen from './screens/ChatScreen';

export default function App(): JSX.Element {
  console.log('[App] Rendering App component');

  return <ChatScreen />;
}
