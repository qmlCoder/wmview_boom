import Sider from './Plugin.vue'
import icon from './icon.png'
import apidoc from './plugin-api.d.ts?raw'
import type { Plugin } from './plugin'

/**
 * 插件入口（与主程序约定）：
 * - vite.config.ts 的 exposes['./index'] 指向本文件；
 * - 必须同时导出 `plugin`（生产/本地安装加载）与 `default`（开发模式加载）。
 */
export const plugin: Plugin = {
  name: 'boom', // 侧边栏/日志里显示的名字
  sider: { name: 'boom', icon, comp: Sider },
  position: 'right',
  // 推理在本进程内的 Rust→wasm 里跑（不再依赖外部 python 运行时），Windows 与 Android 都能用
  device: ['windows', 'android'],
  apidoc,
}

export default plugin
