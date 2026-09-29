/**
 * boom 推理运行时：加载 Rust→wasm 推理核、扁平权重、固化元素表与 Q/D/p 系数，
 * 之后预测全程在**本进程内**完成（没有外部进程、没有 Python、不联网）。
 *
 * 资源随插件包发布，用 Vite 的 `?url` 拿到地址再 fetch：插件在宿主里由
 * `http://asset.localhost/plugs/wmview.boom/` 提供，相对地址跟着产物目录解析，
 * 所以 dev server 与安装版用同一套代码。
 *
 * ⚠️ 这条链路要求 `vite.config.ts` 里 **`base: './'`**：默认的 `base: '/'` 会把
 * `?url` 资源编译成绝对的 `/assets/...`，在宿主里会解析到主机根目录而 404。
 * （这也是原 Python 版 vite 注释里「外链 assets 取不到」的真正原因。）
 */

import wasmUrl from '../../boom-core/pkg/boom_core_bg.wasm?url'
import tableUrl from '../../boom-data/element_table.json?url'
import densityIndexUrl from '../../boom-data/density.json?url'
import densityBinUrl from '../../boom-data/density.bin?url'
import eofIndexUrl from '../../boom-data/eof.json?url'
import eofBinUrl from '../../boom-data/eof.bin?url'
import perfCoeffUrl from '../../boom-data/perf_coeff.json?url'
import kamletCoeffUrl from '../../boom-data/perf_coeff_kamlet.json?url'

import {
  describe as wasmDescribe,
  initSync,
  is_ready,
  load_model,
  load_perf,
  load_table,
  predict as wasmPredict,
} from '../../boom-core/pkg/boom_core.js'

import type { BoomResult } from '../plugin-api'

/** 推理核自检信息（`describe()` 的 JSON 解析结果）。 */
export interface BoomDescribe {
  ready: boolean
  hidden?: number
  in_dim?: number
  pass_layers?: number
  tensors_per_model?: number
  params_per_model?: number
  density?: { epoch?: number; val_mae?: number; label?: string }
  eof?: { epoch?: number; val_mae?: number; label?: string }
}

export interface BoomRuntime {
  /** 各资源字节数，便于在面板上确认包是否完整 */
  sizes: { wasm: number; weights: number }
  /** 首次加载耗时（毫秒） */
  loadMs: number
  /** 自检信息 */
  info: BoomDescribe
  /** 单分子预测（同步：wasm 里跑完就返回） */
  predict: (request: Record<string, unknown>) => BoomResult
}

/** 加载进度回调（面板直接显示）。 */
export type LoadProgress = (message: string) => void

let runtimePromise: Promise<BoomRuntime> | null = null
let loaded: BoomRuntime | null = null

/** 已加载完成的运行时（未加载完返回 null）。 */
export function boomRuntime(): BoomRuntime | null {
  return loaded
}

async function fetchBytes(url: string, what: string, report: LoadProgress): Promise<ArrayBuffer> {
  report(`下载 ${what}…`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${what} 加载失败（HTTP ${res.status}）：${url}`)
  return res.arrayBuffer()
}

async function fetchText(url: string, what: string, report: LoadProgress): Promise<string> {
  report(`下载 ${what}…`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${what} 加载失败（HTTP ${res.status}）：${url}`)
  return res.text()
}

async function buildRuntime(report: LoadProgress): Promise<BoomRuntime> {
  const t0 = performance.now()
  report('加载推理核与固化表…')

  // 索引/系数是几十 KB 的文本，权重与 wasm 是二进制：一起发出去，靠浏览器并发
  const [wasmBytes, tableJson, dIdx, dBin, eIdx, eBin, perfJson, kamletJson] = await Promise.all([
    fetchBytes(wasmUrl, '推理核 wasm', report),
    fetchText(tableUrl, '元素描述子表', report),
    fetchText(densityIndexUrl, '密度模型索引', report),
    fetchBytes(densityBinUrl, '密度模型权重（14.5 MB）', report),
    fetchText(eofIndexUrl, '生成焓模型索引', report),
    fetchBytes(eofBinUrl, '生成焓模型权重（14.5 MB）', report),
    fetchText(perfCoeffUrl, 'Q/D/p 系数', report),
    fetchText(kamletCoeffUrl, 'Kamlet 系数', report),
  ])

  report('实例化推理核…')
  // 手工实例化：地址由构建时决定，不依赖 wasm-bindgen 自动定位（宿主里没有 host 根路径）
  initSync({ module: new Uint8Array(wasmBytes) })
  load_table(tableJson)
  const nParam = load_model('density', dIdx, new Uint8Array(dBin))
  load_model('eof', eIdx, new Uint8Array(eBin))
  load_perf(perfJson, kamletJson)
  if (!is_ready()) throw new Error('推理核未就绪：权重/表/系数没有全部加载成功')

  const info = JSON.parse(wasmDescribe()) as BoomDescribe
  const runtime: BoomRuntime = {
    sizes: { wasm: wasmBytes.byteLength, weights: dBin.byteLength + eBin.byteLength },
    loadMs: performance.now() - t0,
    info,
    predict: (request) => JSON.parse(wasmPredict(JSON.stringify(request))) as BoomResult,
  }
  if (info.params_per_model && info.params_per_model !== nParam) {
    throw new Error(`权重自检不一致：load_model 返回 ${nParam}，describe 报告 ${info.params_per_model}`)
  }
  loaded = runtime
  return runtime
}

/**
 * 加载运行时（幂等：多处并发调用只会真正加载一次）。
 * 失败时清空缓存，便于用户重试。
 */
export function loadBoomRuntime(report: LoadProgress = () => {}): Promise<BoomRuntime> {
  if (!runtimePromise) {
    runtimePromise = buildRuntime(report).catch((err: unknown) => {
      runtimePromise = null
      loaded = null
      throw err
    })
  }
  return runtimePromise
}
