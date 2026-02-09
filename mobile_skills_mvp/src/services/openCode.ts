/**
 * OpenCode HTTP API Service
 * 用于调用OpenCode Server的RESTful API
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';

// OpenCode Server地址
// 在Windows本地测试时：http://localhost:4096
// 在Mac mini或通过公网IP访问时：修改为实际IP
const BASE_URL = 'http://localhost:4096';

// API接口类型定义
export interface MessagePart {
  type: 'text' | 'step-start' | 'reasoning' | 'step-finish';
  text?: string;
}

export interface MessageInfo {
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
}

export interface Message {
  info: MessageInfo;
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

export interface CreateSessionRequest {
  title?: string;
}

export interface CreateSessionResponse {
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

export interface SendMessageRequest {
  messageID?: string;
  model?: string;
  agent?: string;
  noReply?: boolean;
  system?: string;
  tools?: any[];
  parts: MessagePart[];
}

export interface SendMessageResponse {
  info: MessageInfo;
  parts: MessagePart[];
}

export interface GetMessagesResponse {
  info: MessageInfo[];
  parts: MessagePart[];
}

export interface HealthCheckResponse {
  healthy: boolean;
  version: string;
}

/**
 * OpenCode API Service
 */
export class OpenCodeService {
  private axiosInstance: AxiosInstance;
  private sessionId: string | null = null;

  constructor(baseURL: string = BASE_URL) {
    this.axiosInstance = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * 健康检查
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    try {
      const response = await this.axiosInstance.get<HealthCheckResponse>('/global/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  /**
   * 创建新会话
   */
  async createSession(title: string = 'New Chat'): Promise<CreateSessionResponse> {
    try {
      const response = await this.axiosInstance.post<CreateSessionResponse>('/session', {
        title,
      });
      this.sessionId = response.data.id;
      return response.data;
    } catch (error) {
      console.error('Create session failed:', error);
      throw error;
    }
  }

  /**
   * 获取会话详情
   */
  async getSession(sessionId: string): Promise<Session> {
    try {
      const response = await this.axiosInstance.get<Session>(`/session/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Get session failed:', error);
      throw error;
    }
  }

  /**
   * 发送消息
   */
  async sendMessage(
    content: string,
    sessionId?: string
  ): Promise<SendMessageResponse> {
    try {
      const id = sessionId || this.sessionId;
      if (!id) {
        throw new Error('No session ID');
      }

      const request: SendMessageRequest = {
        parts: [
          {
            type: 'text',
            text: content,
          },
        ],
      };

      const response = await this.axiosInstance.post<SendMessageResponse>(
        `/session/${id}/message`,
        request
      );

      return response.data;
    } catch (error) {
      console.error('Send message failed:', error);
      throw error;
    }
  }

  /**
   * 获取消息列表
   */
  async getMessages(sessionId?: string): Promise<GetMessagesResponse> {
    try {
      const id = sessionId || this.sessionId;
      if (!id) {
        throw new Error('No session ID');
      }

      const response = await this.axiosInstance.get<GetMessagesResponse>(
        `/session/${id}/message`
      );

      return response.data;
    } catch (error) {
      console.error('Get messages failed:', error);
      throw error;
    }
  }

  /**
   * 删除会话
   */
  async deleteSession(sessionId?: string): Promise<boolean> {
    try {
      const id = sessionId || this.sessionId;
      if (!id) {
        throw new Error('No session ID');
      }

      const response = await this.axiosInstance.delete<boolean>(`/session/${id}`);

      if (id === this.sessionId) {
        this.sessionId = null;
      }

      return response.data;
    } catch (error) {
      console.error('Delete session failed:', error);
      throw error;
    }
  }

  /**
   * 设置当前会话ID
   */
  setSessionId(sessionId: string) {
    this.sessionId = sessionId;
  }

  /**
   * 获取当前会话ID
   */
  getSessionId(): string | null {
    return this.sessionId;
  }
}

export default new OpenCodeService();
