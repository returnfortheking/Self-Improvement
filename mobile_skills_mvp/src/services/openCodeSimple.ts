/**
 * OpenCode HTTP API Service (使用 fetch API - 更好的 React Native 兼容性)
 */

// 简单的缓存，避免频繁请求
const cache = {
  healthCheck: null as any,
  sessionInfo: null as any,
  messages: [] as any[],
};

// API接口类型定义（简化版）
export interface SimpleMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  time?: number;
}

export interface SimpleSession {
  id: string;
  title: string;
}

export class OpenCodeServiceSimple {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:4096') {
    this.baseUrl = baseUrl;
    console.log(`[OpenCodeService] Initialized with baseUrl: ${baseUrl}`);
  }

  /**
   * 通用的 fetch 方法
   */
  private async fetch(endpoint: string, options?: RequestInit): Promise<any> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      console.log(`[OpenCodeService] ${options?.method || 'GET'} ${url}`);

      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`[OpenCodeService] Response:`, data);
      return data;
    } catch (error: any) {
      console.error(`[OpenCodeService] Request failed:`, error);
      throw error;
    }
  }

  /**
   * 简化的健康检查
   */
  async healthCheck(): Promise<{ healthy: boolean; version: string }> {
    // 使用缓存
    if (cache.healthCheck) {
      return cache.healthCheck;
    }

    try {
      console.log('[Service] Checking health...');
      const response = await this.fetch('/global/health');

      cache.healthCheck = response;
      console.log('[Service] Health check:', response);
      return response;
    } catch (error) {
      console.error('[Service] Health check failed:', error);
      throw error;
    }
  }

  /**
   * 简化的创建session
   */
  async createSession(title: string = 'Chat'): Promise<SimpleSession> {
    try {
      console.log('[Service] Creating session with title:', title);

      const response = await this.fetch('/session', {
        method: 'POST',
        body: JSON.stringify({ title }),
      });

      console.log('[Service] Session created:', response);
      return response;
    } catch (error) {
      console.error('[Service] Create session failed:', error);
      throw error;
    }
  }

  /**
   * 简化的发送消息
   */
  async sendMessage(sessionId: string, content: string): Promise<any> {
    try {
      console.log('[Service] Sending message:', content);

      const response = await this.fetch(`/session/${sessionId}/message`, {
        method: 'POST',
        body: JSON.stringify({
          parts: [
            {
              type: 'text',
              text: content,
            },
          ],
        }),
      });

      console.log('[Service] Message sent:', response);

      // 清除缓存
      cache.sessionInfo = null;
      cache.messages = [];

      return response;
    } catch (error) {
      console.error('[Service] Send message failed:', error);
      throw error;
    }
  }

  /**
   * 简化的获取消息列表
   */
  async getMessages(sessionId: string): Promise<{ info: any[]; parts: any[] }> {
    try {
      console.log('[Service] Getting messages...');

      const response = await this.fetch(`/session/${sessionId}/message`);

      console.log('[Service] Raw response type:', Array.isArray(response) ? 'array' : typeof response);
      console.log('[Service] Raw response length:', Array.isArray(response) ? response.length : 'N/A');

      // 响应可能是数组格式：[{ info: ..., parts: ... }, ...]
      // 或者对象格式：{ info: [...], parts: [...] }
      let messages: any[] = [];
      if (Array.isArray(response)) {
        console.log('[Service] Response is array format');
        messages = response;
      } else if (response && Array.isArray(response.info)) {
        console.log('[Service] Response is object format with info array');
        messages = response.info;
      } else {
        console.log('[Service] Response format unrecognized, defaulting to empty array');
      }

      console.log('[Service] Messages array length:', messages.length);

      const simplified = {
        info: messages.map((msgWrapper: any) => {
          // 数组格式：[{ info: {...}, parts: [...] }, ...]
          const msg = msgWrapper.info || msgWrapper;
          const parts = msgWrapper.parts || msg.parts || [];

          // 查找 text 类型的 part（type='text'），如果没有则使用第一个 part
          const textPart = parts.find((p: any) => p.type === 'text') || parts[0];
          const content = textPart?.text || '';

          console.log('[Service] Processing message:', {
            id: msg.id,
            role: msg.role,
            partsCount: parts.length,
            contentPreview: content.substring(0, 50) + (content.length > 50 ? '...' : ''),
          });

          return {
            id: msg.id,
            role: msg.role,
            content: content,
            time: msg.time?.created || Date.now(),
          };
        }),
        parts: messages.map((msgWrapper: any) => msgWrapper.parts || msgWrapper.parts || []),
      };

      // 缓存消息
      cache.messages = simplified.info;
      cache.sessionInfo = response;

      console.log('[Service] Simplified messages count:', simplified.info.length);
      return simplified;
    } catch (error) {
      console.error('[Service] Get messages failed:', error);
      throw error;
    }
  }

  /**
   * 删除session
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      console.log('[Service] Deleting session:', sessionId);

      const response = await this.fetch(`/session/${sessionId}`, {
        method: 'DELETE',
      });

      console.log('[Service] Session deleted:', response);

      // 清除缓存
      cache.sessionInfo = null;
      cache.messages = [];

      return response;
    } catch (error) {
      console.error('[Service] Delete session failed:', error);
      throw error;
    }
  }
}
