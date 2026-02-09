/**
 * ServerManager 类型定义
 */

export interface ServerConfig {
  /** 服务器名称 */
  name: string;
  /** 服务器 URL（ngrok 地址或其他公网地址） */
  url: string;
  /** 优先级（数字越小优先级越高） */
  priority: number;
  /** 是否启用 */
  enabled: boolean;
}

export interface ServerStatus {
  /** 是否健康 */
  healthy: boolean;
  /** 最后检查时间 */
  lastCheck?: number;
  /** 错误信息 */
  error?: string;
  /** 是否可用（用于故障转移） */
  available?: boolean;
  /** 失败次数 */
  failureCount?: number;
  /** 平均响应时间（毫秒） */
  avgResponseTime?: number;
}

export interface ServerHealthResult {
  /** 服务器配置 */
  server: ServerConfig;
  /** 健康状态 */
  status: ServerStatus;
}

export interface ServerManagerConfig {
  /** 服务器列表 */
  servers: ServerConfig[];
  /** 健康检查间隔（毫秒） */
  healthCheckInterval?: number;
  /** 故障服务器恢复超时（毫秒） */
  failureRecoveryTimeout?: number;
  /** 最大失败次数 */
  maxFailures?: number;
}
