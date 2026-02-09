/**
 * ServerManager - 多服务器管理器
 *
 * 功能：
 * - 管理多个 OpenCode Server 配置
 * - 自动健康检查
 * - 优先级排序和故障转移
 * - 配置持久化
 */

import { ServerConfig, ServerStatus, ServerHealthResult, ServerManagerConfig } from './ServerManager.types';

export class ServerManager {
  private servers: ServerConfig[] = [];
  private serverStatusMap: Map<string, ServerStatus> = new Map();
  private healthCheckInterval: number;
  private failureRecoveryTimeout: number;
  private maxFailures: number;

  constructor(config: ServerManagerConfig | ServerConfig[]) {
    if (Array.isArray(config)) {
      // 兼容旧版本：直接传入服务器数组
      this.servers = config;
      this.healthCheckInterval = 60000; // 默认 1 分钟
      this.failureRecoveryTimeout = 5 * 60 * 1000; // 默认 5 分钟
      this.maxFailures = 3;
    } else {
      this.servers = config.servers;
      this.healthCheckInterval = config.healthCheckInterval || 60000;
      this.failureRecoveryTimeout = config.failureRecoveryTimeout || 5 * 60 * 1000;
      this.maxFailures = config.maxFailures || 3;
    }

    // 初始化服务器状态
    this.servers.forEach(server => {
      this.serverStatusMap.set(server.name, {
        healthy: false,
        available: true,
        failureCount: 0,
      });
    });
  }

  /**
   * 获取所有服务器配置
   */
  getServers(): ServerConfig[] {
    return [...this.servers];
  }

  /**
   * 获取启用的服务器
   */
  getEnabledServers(): ServerConfig[] {
    return this.servers.filter(s => s.enabled);
  }

  /**
   * 按优先级排序的服务器
   */
  getServersByPriority(): ServerConfig[] {
    return this.getEnabledServers().sort((a, b) => a.priority - b.priority);
  }

  /**
   * 根据名称获取服务器
   */
  getServer(name: string): ServerConfig | undefined {
    return this.servers.find(s => s.name === name);
  }

  /**
   * 获取服务器状态
   */
  getServerStatus(server: ServerConfig): ServerStatus | undefined {
    return this.serverStatusMap.get(server.name);
  }

  /**
   * 检查单个服务器健康状态
   */
  async checkServerHealth(server: ServerConfig): Promise<ServerStatus> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${server.url}/global/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000), // 10 秒超时
      });

      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        const status: ServerStatus = {
          healthy: data.healthy || true,
          lastCheck: Date.now(),
          available: true,
          failureCount: 0,
          avgResponseTime: responseTime,
        };

        this.serverStatusMap.set(server.name, status);
        return status;
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      const status: ServerStatus = {
        healthy: false,
        lastCheck: Date.now(),
        error: error instanceof Error ? error.message : String(error),
        available: true, // 初始可用，失败后会标记为不可用
        failureCount: 0,
      };

      this.serverStatusMap.set(server.name, status);
      return status;
    }
  }

  /**
   * 检查所有启用的服务器健康状态
   */
  async checkAllServers(): Promise<ServerHealthResult[]> {
    const enabledServers = this.getEnabledServers();
    const results: ServerHealthResult[] = [];

    for (const server of enabledServers) {
      const status = await this.checkServerHealth(server);
      results.push({ server, status });
    }

    return results;
  }

  /**
   * 选择最佳服务器（优先级最高且健康的）
   */
  async selectBestServer(): Promise<ServerConfig | null> {
    const serversByPriority = this.getServersByPriority();

    for (const server of serversByPriority) {
      const status = this.getServerStatus(server);

      // 跳过不可用的服务器
      if (!status?.available) {
        continue;
      }

      // 检查健康状态
      const healthStatus = await this.checkServerHealth(server);

      if (healthStatus.healthy) {
        console.log(`[ServerManager] Selected server: ${server.name}`);
        return server;
      } else {
        // 标记为失败
        await this.handleServerFailure(server);
      }
    }

    console.warn('[ServerManager] No healthy servers available');
    return null;
  }

  /**
   * 处理服务器失败
   */
  async handleServerFailure(server: ServerConfig): Promise<void> {
    const status = this.getServerStatus(server);

    if (!status) return;

    const newFailureCount = (status.failureCount || 0) + 1;

    if (newFailureCount >= this.maxFailures) {
      // 超过最大失败次数，标记为不可用
      console.warn(
        `[ServerManager] Server ${server.name} marked as unavailable after ${newFailureCount} failures`
      );

      this.serverStatusMap.set(server.name, {
        ...status,
        available: false,
        failureCount: newFailureCount,
      });

      // 设置恢复定时器
      setTimeout(() => {
        console.log(`[ServerManager] Attempting to recover server: ${server.name}`);
        this.serverStatusMap.set(server.name, {
          ...status,
          available: true,
          failureCount: 0,
        });
      }, this.failureRecoveryTimeout);
    } else {
      // 更新失败次数
      this.serverStatusMap.set(server.name, {
        ...status,
        failureCount: newFailureCount,
      });
    }
  }

  /**
   * 添加新服务器
   */
  addServer(server: ServerConfig): void {
    if (this.servers.find(s => s.name === server.name)) {
      throw new Error(`Server with name "${server.name}" already exists`);
    }

    this.servers.push(server);
    this.serverStatusMap.set(server.name, {
      healthy: false,
      available: true,
      failureCount: 0,
    });
  }

  /**
   * 更新服务器配置
   */
  updateServer(server: ServerConfig): void {
    const index = this.servers.findIndex(s => s.name === server.name);

    if (index === -1) {
      throw new Error(`Server "${server.name}" not found`);
    }

    this.servers[index] = server;
  }

  /**
   * 删除服务器
   */
  removeServer(name: string): void {
    const index = this.servers.findIndex(s => s.name === name);

    if (index === -1) {
      throw new Error(`Server "${name}" not found`);
    }

    this.servers.splice(index, 1);
    this.serverStatusMap.delete(name);
  }

  /**
   * 导出配置为 JSON
   */
  exportConfig(): { servers: ServerConfig[] } {
    return {
      servers: this.getServers(),
    };
  }

  /**
   * 从 JSON 导入配置
   */
  importConfig(config: { servers: ServerConfig[] }): void {
    this.servers = config.servers;
    this.serverStatusMap.clear();

    // 重新初始化状态
    this.servers.forEach(server => {
      this.serverStatusMap.set(server.name, {
        healthy: false,
        available: true,
        failureCount: 0,
      });
    });
  }

  /**
   * 启动自动健康检查
   */
  startHealthCheck(): NodeJS.Timeout {
    return setInterval(async () => {
      console.log('[ServerManager] Running health check...');
      await this.checkAllServers();
    }, this.healthCheckInterval);
  }
}
