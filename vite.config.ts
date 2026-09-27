import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { federation } from '@module-federation/vite'

export default defineConfig({
  plugins: [
    vue(),
    federation({
      // 与安装目录 wmview.boom 对应（主程序 src/plugin/mf.ts 的 remoteNameOf）
      name: 'wmview_boom',
      filename: 'index.js',
      exposes: {
        './index': './src/index.ts',
      },
      bundleAllCSS: true,
      dts: false,
      // vue / element-plus / three 由主程序（MF Host）以 singleton 提供
      shared: {
        vue: { singleton: true, requiredVersion: false },
        'element-plus': { singleton: true, requiredVersion: false },
        three: { singleton: true, requiredVersion: false },
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // 主程序开发模式固定连 http://localhost:3001/index.js
    port: 3001,
    strictPort: true,
    cors: true,
    hmr: true,
    origin: 'http://localhost:3001',
  },
  build: {
    minify: false,
    // 图标等小资源内联成 data URL：插件是从宿主 plugs 目录加载的，
    // 外链 assets 路径在宿主里取不到（主程序把 icon 原样显示）
    assetsInlineLimit: 200000,
  },
})
