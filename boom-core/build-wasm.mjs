// 构建 wasm 包（`npm run build:wasm`）。
//
// 为什么不用 `.cargo/config.toml` 里的 `[target.wasm32-unknown-unknown] rustflags`：
// cargo 的 RUSTFLAGS 与 config 的 rustflags 是**互斥来源**，wasm-pack 自己会设置
// RUSTFLAGS，于是 config 里的那一条被忽略（实测：二进制字节完全相同）。
// 所以这里显式把 `-C target-feature=+simd128` 并进 RUSTFLAGS 再调 wasm-pack。
//
// 不开 SIMD 的代价很实在：GRMP 前向 ~90% 的算力在矩阵乘，标量 wasm 比原生慢约 10 倍。
// 要求宿主 WebView 支持 wasm SIMD（WebView2 / Android System WebView 自 Chromium 91 起支持）。
import { spawnSync } from 'node:child_process'

const SIMD = '-C target-feature=+simd128'
process.env.RUSTFLAGS = process.env.RUSTFLAGS ? `${process.env.RUSTFLAGS} ${SIMD}` : SIMD

const r = spawnSync(
  'wasm-pack',
  ['build', 'boom-core', '--release', '--target', 'web', '--out-dir', 'pkg'],
  { stdio: 'inherit', shell: true },
)
if (r.error) {
  console.error('调不起 wasm-pack：', r.error.message)
  process.exit(1)
}
process.exit(r.status ?? 1)
