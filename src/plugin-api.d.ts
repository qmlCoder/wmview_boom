/**
 * wmview.boom 对外提供的 API（运行期挂在 window.wmapi_boom 上，见 src/Plugin.vue）。
 *
 * 依赖：已准备好插件目录下的内置 python 运行时（python tools/prepare_python.py）
 * 以及 runtime/weights 下的权重与固化表。
 */

export interface BoomDerived {
  /** 爆热 Q (cal/g)，经验估算 */
  Q: number
  /** 爆速 D (km/s)，经验估算 */
  D: number
  /** 爆压 p (GPa)，经验估算 */
  p: number
}

export interface BoomResult {
  ok: boolean
  /** 本次运行令牌（前端用来确认结果文件不是上一次运行的残留） */
  token?: string
  /** 分子名（分子文件夹下的文件名） */
  name: string
  /** Hill 式分子式 */
  formula: string
  /** 原子数 / 重原子数 */
  n_atoms: number
  n_heavy: number
  /** 建模用的键图来源：host（宿主化学键）或 distance（距离判据） */
  bond_mode: string
  /** 模型输出：密度 (g/cm³) 与固体生成焓 (kcal/mol) */
  results: { density?: number; eof?: number }
  /** Q / D / p 的经验估算（不是模型输出），未开启时为 null */
  derived: BoomDerived | null
  /** 超出训练集覆盖域等提示 */
  warnings: string[]
  /** ok=false 时的错误信息 */
  error?: string
}

export interface wmapi_boom {
  /**
   * 用 GRMP 模型预测一个分子
   * @param mole_name 分子文件夹下的文件名（见 wmapi_files.get_file_list）
   * @param targets 要跑的模型，默认两个都跑：'density'（密度）/ 'eof'（固体生成焓）
   * @returns 是否成功，以及结果 JSON 文本（结构见 BoomResult）
   */
  predict(mole_name: string, targets?: ('density' | 'eof')[]): Promise<[boolean, string]>
  /**
   * 同上，但直接返回结构化结果
   */
  predict_result(mole_name: string, targets?: ('density' | 'eof')[]): Promise<BoomResult>
}

declare global {
  interface Window {
    wmapi_boom: wmapi_boom
  }
}
