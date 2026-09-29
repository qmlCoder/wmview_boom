// 验证 wasm 链路：加载 pkg/ 里的 wasm + 扁平权重 + 固化表，喂 tools/golden.json 里的
// 分子（参考值是**原始 PyTorch** 的输出），比较偏差并测单分子耗时。
//
// 与原生 CLI 的 --golden 是同一个门禁，区别在于这条路径会真正走一遍 wasm 边界
// （UTF-8 字符串进出、Uint8Array 权重、JSON 序列化）。
//
// 用法: node boom-core/verify-wasm.mjs
import fs from 'node:fs'
import {
  initSync, load_table, load_model, load_perf, is_ready, describe, predict,
} from './pkg/boom_core.js'

const ROOT = new URL('../', import.meta.url)
const readText = (p) => fs.readFileSync(new URL(p, ROOT), 'utf8')
const readBin = (p) => fs.readFileSync(new URL(p, ROOT))
const TOL = 1e-4

const t0 = performance.now()
initSync({ module: readBin('boom-core/pkg/boom_core_bg.wasm') })
load_table(readText('boom-data/element_table.json'))
load_model('density', readText('boom-data/density.json'), new Uint8Array(readBin('boom-data/density.bin')))
load_model('eof', readText('boom-data/eof.json'), new Uint8Array(readBin('boom-data/eof.bin')))
load_perf(readText('boom-data/perf_coeff.json'), readText('boom-data/perf_coeff_kamlet.json'))
const loadMs = performance.now() - t0
console.log(`wasm 初始化 + 权重加载完成：${loadMs.toFixed(0)} ms`)
console.log('自检摘要：' + describe())
if (!is_ready()) throw new Error('推理核未就绪')

const mols = JSON.parse(readText('tools/golden.json'))
console.log(`\n=== 与原始 PyTorch 参考输出对拍（${mols.length} 个分子 × 2 个模型） ===`)
let worst = 0
let worstAt = ''
for (const m of mols) {
  const json = predict(JSON.stringify({
    name: m.name, syms: m.syms, xyzs: m.xyzs, bonds: m.bonds,
    bond_mode: 'host', index_mode: 'legacy', targets: ['density', 'eof'], derived: true,
  }))
  const r = JSON.parse(json)
  if (!r.ok) throw new Error(`${m.name} 预测失败：${r.error}`)
  const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-6)
  const rd = rel(r.results.density, m.density)
  const re = rel(r.results.eof, m.eof)
  for (const [tag, v] of [['density', rd], ['eof', re]]) {
    if (v > worst) { worst = v; worstAt = `${m.name} ${tag}` }
  }
  console.log(`  ${m.name.padStart(16)}  密度 ${r.results.density.toFixed(6)} (torch ${m.density.toFixed(6)}, ` +
    `相对 ${rd.toExponential(2)})  ΔHf ${r.results.eof.toFixed(6)} (torch ${m.eof.toFixed(6)}, 相对 ${re.toExponential(2)})` +
    `  Q=${r.derived.Q.toFixed(1)} D=${r.derived.D.toFixed(3)} p=${r.derived.p.toFixed(2)}`)
}
console.log(`\n最大相对偏差 ${worst.toExponential(3)}（阈值 ${TOL.toExponential(0)}，最大处：${worstAt}）`)

// 单分子耗时：用最大的那个分子反复跑
const big = mols.reduce((a, b) => (a.syms.length > b.syms.length ? a : b))
const payload = JSON.stringify({
  name: big.name, syms: big.syms, xyzs: big.xyzs, bonds: big.bonds,
  bond_mode: 'host', index_mode: 'legacy', targets: ['density', 'eof'], derived: true,
})
predict(payload)
const N = 200
const t1 = performance.now()
for (let i = 0; i < N; i++) predict(payload)
const per = (performance.now() - t1) / N
console.log(`单分子前向（两个模型 + Q/D/p）：${per.toFixed(3)} ms/分子（${big.name}，${big.syms.length} 原子）`)
console.log(`首次加载 ${loadMs.toFixed(0)} ms`)

console.log('\n=== 结论 ===')
const ok = worst < TOL
console.log(`最大相对偏差：${worst.toExponential(3)}`)
console.log(`判定：${ok ? '通过' : '不通过'}`)
process.exit(ok ? 0 : 1)
