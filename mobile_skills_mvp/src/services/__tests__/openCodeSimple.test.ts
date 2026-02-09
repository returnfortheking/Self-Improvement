/**
 * OpenCodeServiceSimple 单元测试
 *
 * 测试API服务的核心功能：
 * - 健康检查
 * - 创建会话
 * - 发送消息
 * - 获取消息列表
 * - 删除会话
 */

import { OpenCodeServiceSimple } from '../openCodeSimple';

// Mock fetch API
global.fetch = jest.fn();

describe('OpenCodeServiceSimple', () => {
  let service: OpenCodeServiceSimple;
  const mockBaseUrl = 'http://localhost:4096';

  beforeEach(() => {
    service = new OpenCodeServiceSimple(mockBaseUrl);
    (global.fetch as jest.Mock).mockClear();
  });

  describe('healthCheck', () => {
    it('should return health status when API responds successfully', async () => {
      const mockResponse = {
        healthy: true,
        version: '1.0.0',
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await service.healthCheck();

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/global/health`,
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('should throw error when API request fails', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      } as Response);

      await expect(service.healthCheck()).rejects.toThrow('HTTP 500: Internal Server Error');
    });
  });

  describe('createSession', () => {
    it('should create a new session with title', async () => {
      const mockSession = {
        id: 'ses_test123',
        title: 'Test Session',
        created: Date.now(),
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockSession,
      } as Response);

      const result = await service.createSession('Test Session');

      expect(result.id).toBe(mockSession.id);
      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/session`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ title: 'Test Session' }),
        })
      );
    });

    it('should use default title when not provided', async () => {
      const mockSession = {
        id: 'ses_test456',
        title: 'Chat',
        created: Date.now(),
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockSession,
      } as Response);

      await service.createSession();

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/session`,
        expect.objectContaining({
          body: JSON.stringify({ title: 'Chat' }),
        })
      );
    });
  });

  describe('sendMessage', () => {
    it('should send message with correct format', async () => {
      const sessionId = 'ses_test123';
      const content = 'Test message';

      const mockResponse = {
        info: { id: 'msg_test123', role: 'assistant' },
        parts: [{ type: 'text', text: 'Response' }],
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await service.sendMessage(sessionId, content);

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/session/${sessionId}/message`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            parts: [{ type: 'text', text: content }],
          }),
        })
      );
    });
  });

  describe('getMessages', () => {
    it('should parse array format response correctly', async () => {
      const sessionId = 'ses_test123';

      const mockArrayResponse = [
        {
          info: {
            id: 'msg_user1',
            role: 'user',
            time: { created: Date.now() },
          },
          parts: [{ type: 'text', text: 'Hello' }],
        },
        {
          info: {
            id: 'msg_assistant1',
            role: 'assistant',
            time: { created: Date.now() },
          },
          parts: [
            { type: 'step-start', snapshot: 'abc123' },
            { type: 'text', text: 'Hi there!' },
          ],
        },
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockArrayResponse,
      } as Response);

      const result = await service.getMessages(sessionId);

      expect(result.info).toHaveLength(2);
      expect(result.info[0]).toMatchObject({
        id: 'msg_user1',
        role: 'user',
        content: 'Hello',
      });
      expect(result.info[1]).toMatchObject({
        id: 'msg_assistant1',
        role: 'assistant',
        content: 'Hi there!',
      });
    });

    it('should extract text part from multiple parts', async () => {
      const sessionId = 'ses_test123';

      const mockResponse = [
        {
          info: {
            id: 'msg_test',
            role: 'assistant',
            time: { created: Date.now() },
          },
          parts: [
            { type: 'step-start', snapshot: 'abc' },
            { type: 'reasoning', text: 'Thinking...' },
            { type: 'text', text: 'Final answer' },
            { type: 'step-finish', reason: 'stop' },
          ],
        },
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await service.getMessages(sessionId);

      expect(result.info[0].content).toBe('Final answer');
    });

    it('should handle empty response', async () => {
      const sessionId = 'ses_test123';

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => [],
      } as Response);

      const result = await service.getMessages(sessionId);

      expect(result.info).toHaveLength(0);
    });

    it('should handle missing text part gracefully', async () => {
      const sessionId = 'ses_test123';

      const mockResponse = [
        {
          info: {
            id: 'msg_test',
            role: 'assistant',
            time: { created: Date.now() },
          },
          parts: [{ type: 'step-start', snapshot: 'abc' }],
        },
      ];

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const result = await service.getMessages(sessionId);

      expect(result.info[0].content).toBe('');
    });
  });

  describe('deleteSession', () => {
    it('should delete session by ID', async () => {
      const sessionId = 'ses_test123';

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      } as Response);

      const result = await service.deleteSession(sessionId);

      expect(global.fetch).toHaveBeenCalledWith(
        `${mockBaseUrl}/session/${sessionId}`,
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });
  });
});
