/**
 * ServerManager 单元测试
 *
 * 测试多服务器管理功能：
 * - 服务器配置
 * - 健康检查
 * - 优先级排序
 * - 故障转移
 * - 自动选择最佳服务器
 */

import { ServerManager, ServerConfig, ServerStatus } from '../ServerManager';

// Mock fetch API
global.fetch = jest.fn();

describe('ServerManager', () => {
  let serverManager: ServerManager;

  // 测试用的服务器配置
  const mockServers: ServerConfig[] = [
    {
      name: 'Mac Mini',
      url: 'https://mac-mini.ngrok.dev',
      priority: 1, // 高优先级
      enabled: true,
    },
    {
      name: 'Windows PC',
      url: 'https://windows-pc.ngrok.dev',
      priority: 2, // 低优先级
      enabled: true,
    },
    {
      name: 'Backup Server',
      url: 'https://backup.ngrok.dev',
      priority: 3,
      enabled: false, // 禁用
    },
  ];

  beforeEach(() => {
    serverManager = new ServerManager(mockServers);
    (global.fetch as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.clearAllTimers();
  });

  describe('构造函数和初始化', () => {
    it('应该正确初始化服务器列表', () => {
      const servers = serverManager.getServers();
      expect(servers).toHaveLength(3);
    });

    it('应该只返回启用的服务器', () => {
      const enabledServers = serverManager.getEnabledServers();
      expect(enabledServers).toHaveLength(2);
      expect(enabledServers.every(s => s.enabled)).toBe(true);
    });

    it('应该按优先级排序服务器', () => {
      const servers = serverManager.getServersByPriority();
      expect(servers[0].name).toBe('Mac Mini');
      expect(servers[1].name).toBe('Windows PC');
    });
  });

  describe('健康检查', () => {
    it('应该正确检测健康的服务器', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ healthy: true }),
      } as Response);

      const status = await serverManager.checkServerHealth(mockServers[0]);

      expect(status.healthy).toBe(true);
      expect(status.lastCheck).toBeDefined();
    });

    it('应该正确检测不健康的服务器', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(
        new Error('Network error')
      );

      const status = await serverManager.checkServerHealth(mockServers[0]);

      expect(status.healthy).toBe(false);
      expect(status.error).toBe('Network error');
    });

    it('应该检查所有启用的服务器', async () => {
      // Mock: Mac Mini 健康，Windows PC 不健康
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ healthy: true }),
        } as Response)
        .mockRejectedValueOnce(new Error('Timeout'));

      const results = await serverManager.checkAllServers();

      expect(results).toHaveLength(2);
      expect(results[0].server.name).toBe('Mac Mini');
      expect(results[0].status.healthy).toBe(true);
      expect(results[1].server.name).toBe('Windows PC');
      expect(results[1].status.healthy).toBe(false);
    });
  });

  describe('服务器选择', () => {
    it('应该返回第一个健康的服务器', async () => {
      // Mac Mini 健康
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ healthy: true }),
      } as Response);

      const server = await serverManager.selectBestServer();

      expect(server?.name).toBe('Mac Mini');
    });

    it('当首选服务器不健康时，应该返回备用服务器', async () => {
      // Mac Mini 不健康，Windows PC 健康
      (global.fetch as jest.Mock)
        .mockRejectedValueOnce(new Error('Mac Mini down'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ healthy: true }),
        } as Response);

      const server = await serverManager.selectBestServer();

      expect(server?.name).toBe('Windows PC');
    });

    it('当所有服务器都不健康时，应该返回 null', async () => {
      (global.fetch as jest.Mock).mockRejectedValue(
        new Error('All servers down')
      );

      const server = await serverManager.selectBestServer();

      expect(server).toBeNull();
    });

    it('应该跳过禁用的服务器', async () => {
      const disabledManager = new ServerManager([
        mockServers[2], // Backup Server (disabled)
      ]);

      const server = await disabledManager.selectBestServer();

      expect(server).toBeNull();
    });
  });

  describe('故障转移', () => {
    it('应该在当前服务器失败时自动切换到备用服务器', async () => {
      // 初始：Mac Mini 健康
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({ healthy: true }),
      } as Response);

      const initialServer = await serverManager.selectBestServer();
      expect(initialServer?.name).toBe('Mac Mini');

      // Mac Mini 失败，Windows PC 健康
      await serverManager.handleServerFailure(initialServer!);

      const fallbackServer = await serverManager.selectBestServer();
      expect(fallbackServer?.name).toBe('Windows PC');
    });

    it('应该标记失败的服务器为临时不可用', async () => {
      const server = mockServers[0];

      await serverManager.handleServerFailure(server);

      const status = serverManager.getServerStatus(server);
      expect(status?.available).toBe(false);
      expect(status?.failureCount).toBe(1);
    });

    it('应该在超时后恢复失败的服务器', async () => {
      jest.useFakeTimers();

      const server = mockServers[0];
      await serverManager.handleServerFailure(server);

      // 等待恢复超时（5分钟）
      jest.advanceTimersByTime(5 * 60 * 1000);

      const status = serverManager.getServerStatus(server);
      expect(status?.available).toBe(true);

      jest.useRealTimers();
    });
  });

  describe('配置管理', () => {
    it('应该能够添加新服务器', () => {
      const newServer: ServerConfig = {
        name: 'New Server',
        url: 'https://new.ngrok.dev',
        priority: 1,
        enabled: true,
      };

      serverManager.addServer(newServer);

      const servers = serverManager.getServers();
      expect(servers).toHaveLength(4);
      expect(servers.find(s => s.name === 'New Server')).toBeDefined();
    });

    it('应该能够更新服务器配置', () => {
      const updatedServer: ServerConfig = {
        ...mockServers[0],
        enabled: false,
      };

      serverManager.updateServer(updatedServer);

      const server = serverManager.getServer(mockServers[0].name);
      expect(server?.enabled).toBe(false);
    });

    it('应该能够删除服务器', () => {
      serverManager.removeServer('Backup Server');

      const servers = serverManager.getServers();
      expect(servers).toHaveLength(2);
      expect(servers.find(s => s.name === 'Backup Server')).toBeUndefined();
    });
  });

  describe('持久化配置', () => {
    it('应该能够导出配置为 JSON', () => {
      const config = serverManager.exportConfig();

      expect(config).toHaveProperty('servers');
      expect(config.servers).toHaveLength(3);
    });

    it('应该能够从 JSON 导入配置', () => {
      const newConfig: ServerConfig[] = [
        {
          name: 'Imported Server',
          url: 'https://imported.ngrok.dev',
          priority: 1,
          enabled: true,
        },
      ];

      serverManager.importConfig({ servers: newConfig });

      const servers = serverManager.getServers();
      expect(servers).toHaveLength(1);
      expect(servers[0].name).toBe('Imported Server');
    });
  });

  describe('边缘情况', () => {
    it('应该处理空的服务器列表', async () => {
      const emptyManager = new ServerManager([]);

      const server = await emptyManager.selectBestServer();

      expect(server).toBeNull();
    });

    it('应该处理所有服务器都禁用的情况', async () => {
      const allDisabled = mockServers.map(s => ({ ...s, enabled: false }));
      const disabledManager = new ServerManager(allDisabled);

      const server = await disabledManager.selectBestServer();

      expect(server).toBeNull();
    });

    it('应该处理相同优先级的服务器', async () => {
      const samePriority: ServerConfig[] = [
        {
          name: 'Server A',
          url: 'https://a.ngrok.dev',
          priority: 1,
          enabled: true,
        },
        {
          name: 'Server B',
          url: 'https://b.ngrok.dev',
          priority: 1,
          enabled: true,
        },
      ];

      const samePriorityManager = new ServerManager(samePriority);

      const serversByPriority = samePriorityManager.getServersByPriority();

      // 应该保持原始顺序
      expect(serversByPriority[0].name).toBe('Server A');
      expect(serversByPriority[1].name).toBe('Server B');
    });
  });
});
