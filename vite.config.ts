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
  // 插件产物被宿主放到 plugs/wmview.boom/ 下加载（asset.localhost），必须是相对 base：
  // 推理核 wasm 与 29 MB 权重是 `?url` 外部资源，绝对路径 /assets/... 会解析到主机根目录而 404。
  base: './',
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
    // 小资源（图标 20 KB）内联成 data URL：宿主直接拿 icon 字段显示，外链取不到；
    // 权重（14.5 MB × 2）远大于阈值，仍按 assets 文件发布、运行时 fetch。
    assetsInlineLimit: 200000,
  },
})
