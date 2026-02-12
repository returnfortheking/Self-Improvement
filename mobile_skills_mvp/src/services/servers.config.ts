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
    url: 'https://raquel-tiniest-disrespectfully.ngrok-free.dev',
    priority: 1,
    enabled: true,
  },
  {
    name: 'Windows PC',
    url: 'https://rousingly-childlike-latarsha.ngrok-free.dev',
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
