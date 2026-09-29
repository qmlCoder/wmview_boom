/**
 * wmview.boom 对外提供的 API（运行期挂在 window.wmapi_boom 上，见 src/Plugin.vue）。
 *
 * 依赖：随包发布的 Rust→wasm 推理核（`boom-core/pkg/`）与扁平权重（`boom-data/`）。
 * 不再需要内置 python：推理在本进程内完成，Windows 与 Android 都可用。
 */

export interface BoomDerived {
  /** 爆热 Q (cal/g)，经验估算 */
  Q: number
  /** 爆速 D (km/s)，经验估算 */
  D: number
  /** 爆压 p (GPa)，经验估算 */
  p: number
}

/** Q/D/p 的两套实现（见 tools/fit_perf.py） */
export type BoomPerfVariant = 'linear14' | 'kamlet_noco'

export interface BoomResult {
  ok: boolean
  /** 本次运行的令牌（调用方传入时原样回显；进程内调用只是便于串联日志） */
  token?: string
  /** 分子名（分子文件夹下的文件名） */
  name: string
  /** Hill 式分子式 */
  formula: string
  /** 原子数 / 重原子数（重原子 = 除 H 以外的全部元素） */
  n_atoms: number
  n_heavy: number
  /** 各元素计数 */
  counts?: Record<string, number>
  /** 建模用的键图来源：host（宿主化学键）或 distance（距离判据） */
  bond_mode: string
  /** 建模用的键数 */
  n_bonds: number
  /** 元素描述子索引方式：legacy（与权重一致）/ corrected（对照用） */
  index_mode?: string
  /** 模型输出：密度 (g/cm³) 与固体生成焓 (kcal/mol) */
  results: { density?: number; eof?: number }
  /** Q / D / p 的经验估算（不是模型输出），未开启时为 null */
  derived: BoomDerived | null
  /** 另一套 Q/D/p 实现的结果（对照用） */
  derived_alt?: BoomDerived | null
  /** derived 用的是哪一套 */
  perf_variant?: BoomPerfVariant
  /** 超出训练集覆盖域等提示 */
  warnings: string[]
  /** 模型元信息（epoch / val MAE / 标签） */
  meta?: Record<string, { epoch?: number; val_mae?: number; label: string }>
  /** ok=false 时的错误信息 */
  error?: string
}

export interface BoomPredictOptions {
  /** 要跑的模型，默认两个都跑：'density'（密度）/ 'eof'（固体生成焓） */
  targets?: ('density' | 'eof')[]
  /** 键图来源，默认 'auto'（有宿主键用宿主键，否则距离判据） */
  bond_mode?: 'auto' | 'host' | 'distance'
  /** 元素描述子索引方式，默认 'legacy'（与权重一致） */
  index_mode?: 'legacy' | 'corrected'
  /** 是否估算 Q / D / p，默认 true */
  derived?: boolean
  /** Q/D/p 用哪一套实现，默认 'linear14'（现行口径） */
  perf_variant?: BoomPerfVariant
}

export interface wmapi_boom {
  /**
   * 用 GRMP 模型预测一个分子
   * @param mole_name 分子文件夹下的文件名（见 wmapi_files.get_file_list）
   * @param options 模型开关 / 键图来源 / 元素索引 / Q-D-p 口径
   * @returns 是否成功，以及结果 JSON 文本（结构见 BoomResult）
   */
  predict(mole_name: string, options?: BoomPredictOptions): Promise<[boolean, string]>
  /**
   * 同上，但直接返回结构化结果
   */
  predict_result(mole_name: string, options?: BoomPredictOptions): Promise<BoomResult>
  /**
   * 预加载推理核与权重（约 30 MB，首次约 1~2 秒）。
   * 不调用也行：第一次 predict 会自动加载。
   * @returns 自检信息 JSON 文本（推理核版本、参数量、两个模型的 epoch 与 val MAE）
   */
  load(): Promise<string>
}

declare global {
  interface Window {
    wmapi_boom: wmapi_boom
  }
}
