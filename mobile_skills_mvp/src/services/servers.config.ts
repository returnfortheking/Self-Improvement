/**
 * 服务器配置
 *
 * 默认配置：
 * - Mac Mini (优先级 1) - 主要服务器，24小时运行
 * - Windows PC (优先级 2) - 备用服务器
 */

import { ServerConfig } from './ServerManager.types';

export const DEFAULT_SERVERS: ServerConfig[] = [
  {
    name: 'Mac Mini',
    url: 'https://mac-mini-xxxx.ngrok-free.dev', // 需要替换为实际的 ngrok 地址
    priority: 1,
    enabled: true,
  },
  {
    name: 'Windows PC',
    url: 'https://windows-pc-xxxx.ngrok-free.dev', // 需要替换为实际的 ngrok 地址
    priority: 2,
    enabled: true,
  },
];

/**
 * 开发环境配置
 */
export const DEV_SERVERS: ServerConfig[] = [
  {
    name: 'Localhost',
    url: 'http://localhost:4096',
    priority: 1,
    enabled: true,
  },
];

/**
 * 生产环境配置
 */
export const PROD_SERVERS: ServerConfig[] = DEFAULT_SERVERS;
