# Agent Guidelines

## Commands

- `npm test` - Run Jest tests. Use `--testPathPattern=filename` for single test
- `npm start` - Start Metro bundler
- `npm run android` - Run on Android
- `npm run ios` - Run on iOS
- `npx tsc --noEmit` - Type check

## Code Style

### File Structure
- `src/components/` - Reusable UI components (ChatInput, MessageBubble, LoadingSpinner)
- `src/screens/` - Screen components (ChatScreen)
- `src/services/` - API services (openCode, openCodeSimple)

### Imports
Order: React imports, third-party libraries, type imports (using `import type`), relative imports
```typescript
import React from 'react';
import {View, Text} from 'react-native';
import type {NavigationProp} from '@react-navigation/native';
import Component from './Component';
```

### Components
- Functional components with explicit `JSX.Element` return type
- Props interfaces named `ComponentNameProps`
- Export interfaces and components separately
```typescript
export interface MessageBubbleProps {
  message: Message;
  onPress?: () => void;
}

export default function MessageBubble({message, onPress}: MessageBubbleProps): JSX.Element {
  return <View>...</View>;
}
```

### Types
- TypeScript strict mode enabled
- Use interfaces for data structures
- Union types for constrained strings: `'user' | 'assistant' | 'system'`
- Optional props marked with `?`

### Naming
- Components: PascalCase (`MessageBubble`, `ChatScreen`)
- Props interfaces: `ComponentNameProps`
- Services: PascalCase classes (`OpenCodeService`)
- Constants: UPPER_SNAKE_CASE (`BASE_URL`)
- Variables/Functions: camelCase (`sendMessage`)
- Event handlers: `handle` prefix (`handleSend`, `handlePress`, `handleFocus`)

### State Management
- Use `useState` for local component state
- Use `useEffect` for side effects and initialization
- Use `useRef` for imperative control (TextInput focus)
```typescript
const [messages, setMessages] = useState<Message[]>([]);
const scrollViewRef = useRef<ScrollView>(null);

useEffect(() => {
  initConnection();
}, []);
```

### API Services
- Use axios for HTTP requests
- Create service classes with private axios instances
- Define TypeScript interfaces for all API responses
- Use async/await with try-catch for error handling
```typescript
export class OpenCodeService {
  private axiosInstance: AxiosInstance;
  private sessionId: string | null = null;

  async sendMessage(sessionId: string, content: string): Promise<any> {
    try {
      const response = await this.axiosInstance.post(`/session/${sessionId}/message`, {
        parts: [{type: 'text', text: content}]
      });
      return response.data;
    } catch (error) {
      console.error('[Service] Send message failed:', error);
      throw error;
    }
  }
}
```

### Styling
- Use `StyleSheet.create()` at file end
- Group related styles together
- Color constants: `#007AFF` (primary blue), `#f44336` (error red)
- Shadow styling for elevation on Android

### Error Handling
- Async operations wrapped in try-catch
- Log errors with context: `console.error('[Context] Error:', error)`
- Display user-friendly errors via `Alert.alert()`
- Re-throw after logging
- Use error state for UI feedback

### Comments
- File header comment: `/** * ComponentName - Chinese description */`
- Function JSDoc for complex logic
- Console logs for debugging with context tags: `console.log('[ChatScreen] Session created:', session.id)`

### React Native Specifics
- Use `SafeAreaView` for iOS edge handling
- `KeyboardAvoidingView` for form inputs with platform-specific behavior
- Platform checks: `Platform.OS === 'ios'`
- Use refs for imperative control: `useRef<TextInput>(null)`
- `Keyboard.dismiss()` to close keyboard
- `ActivityIndicator` for loading states

### Alert Dialogs
Use `Alert.alert()` for user confirmations:
```typescript
Alert.alert('确认清除会话', '确定要删除当前会话吗？', [
  {text: '取消', style: 'cancel'},
  {text: '确定', style: 'destructive'}
]);
```

### Styling Patterns
- Message bubbles: maxWidth '80%', conditional backgroundColor for user/assistant
- Input fields: borderRadius 20, padding, placeholderTextColor
- Buttons: padding, borderRadius, backgroundColor, centered text
- Status indicators: green (#4CAF50) for connected, red (#f44336) for disconnected
