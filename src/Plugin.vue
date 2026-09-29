<style module>
div.rowBtns {
  display: flex;
  gap: 8px;
  margin: 10px 10px 0;
}

div.res {
  margin: 10px;
  font-size: smaller;
}

table.res {
  width: calc(100% - 20px);
  border-collapse: collapse;
  font-size: smaller;
}

table.res th,
table.res td {
  padding: 2px 4px;
  border-bottom: 1px solid #eee;
  text-align: left;
  white-space: nowrap;
}

table.res td.warn {
  color: #c92a2a;
  white-space: normal;
}

table.res td.alt {
  color: #1971c2;
  white-space: normal;
}

div.log {
  background-color: #1e1e1e;
  color: #d4d4d4;
  font-family: 'Courier New', Courier, monospace;
  font-size: smaller;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 40vh;
  overflow-y: auto;
  padding: 6px;
  border-radius: 4px;
  margin: 10px 5px;
}
</style>

<template>
  <panel-group title="分子">
    <panel-item title="当前分子">
      <span style="font-size: smaller">{{ refs.moleName || '（未选择）' }}</span>
    </panel-item>
    <panel-item title="读取范围" tip="多选文件优先；没有多选时用当前显示/选中的分子">
      <el-select v-model="refs.source" style="width: 100%">
        <el-option label="选中 / 当前分子" value="select"></el-option>
        <el-option label="文件列表全部" value="all"></el-option>
      </el-select>
    </panel-item>
    <panel-item title="键图来源" tip="模型用 RDKit 的键图训练；这里默认优先用宿主识别的化学键，没有时按共价半径判据建键">
      <el-select v-model="refs.bondMode" style="width: 100%">
        <el-option label="自动（优先宿主键）" value="auto"></el-option>
        <el-option label="宿主化学键" value="host"></el-option>
        <el-option label="距离判据" value="distance"></el-option>
      </el-select>
    </panel-item>
    <panel-item title="元素索引" tip="原实现按 n = 原子序数-1 取描述子行（off-by-one），权重就是在这套特征上训练的；corrected 只作对照">
      <el-select v-model="refs.indexMode" style="width: 100%">
        <el-option label="legacy（与权重一致）" value="legacy"></el-option>
        <el-option label="corrected（对照用）" value="corrected"></el-option>
      </el-select>
    </panel-item>
  </panel-group>

  <panel-group title="模型">
    <panel-item title="密度 ρ">
      <el-switch v-model="refs.useDensity"></el-switch>
    </panel-item>
    <panel-item title="生成焓 ΔHf">
      <el-switch v-model="refs.useEof"></el-switch>
    </panel-item>
    <panel-item title="估算 Q / D / p" tip="经验拟合关系（不是模型输出）：Q 误差约 1%，D/p 约 2~5%">
      <el-switch v-model="refs.derived"></el-switch>
    </panel-item>
    <panel-item title="Q/D/p 口径" tip="linear14：现行 14 特征线性回归；kamlet：D/p 换成无 CO 氧平衡 + Kamlet 对数形式（留出集 D 4.19%→2.31%、p 5.35%→4.62%）。两套同时给出，另一套显示在结果下方">
      <el-select v-model="refs.perfVariant" style="width: 100%">
        <el-option label="linear14（现行）" value="linear14"></el-option>
        <el-option label="kamlet（候选）" value="kamlet_noco"></el-option>
      </el-select>
    </panel-item>
  </panel-group>

  <panel-group title="推理核">
    <panel-item title="状态">
      <span style="font-size: smaller">{{ refs.engine }}</span>
    </panel-item>
    <panel-item title="负荷" tip="首次预测要把 wasm 与 29 MB 权重读进内存；之后每个分子只需几十毫秒">
      <span style="font-size: smaller">{{ refs.engineDetail }}</span>
    </panel-item>
  </panel-group>

  <div :class="$style.rowBtns">
    <el-button :icon="Refresh" :loading="refs.checking" @click="check_env">加载推理核</el-button>
    <el-button type="primary" :icon="VideoPlay" :loading="refs.busy" @click="run">开始预测</el-button>
  </div>

  <div v-if="refs.results.length" :class="$style.res">
    <table :class="$style.res">
      <thead>
        <tr>
          <th>分子</th>
          <th>式</th>
          <th>ρ (g/cm³)</th>
          <th>ΔHf (kcal/mol)</th>
          <th>Q (cal/g)</th>
          <th>D (km/s)</th>
          <th>p (GPa)</th>
        </tr>
      </thead>
      <tbody>
        <template v-for="(r, k) in refs.results" :key="k">
          <tr v-if="r.ok">
            <td>{{ r.name }}</td>
            <td>{{ r.formula }}</td>
            <td>{{ fmt(r.results.density, 4) }}</td>
            <td>{{ fmt(r.results.eof, 2) }}</td>
            <td>{{ fmt(r.derived?.Q, 1) }}</td>
            <td>{{ fmt(r.derived?.D, 3) }}</td>
            <td>{{ fmt(r.derived?.p, 2) }}</td>
          </tr>
          <tr v-else>
            <td>{{ r.name }}</td>
            <td :class="$style.warn" colspan="6">{{ r.error }}</td>
          </tr>
          <tr v-if="r.ok && r.warnings?.length">
            <td :class="$style.warn" colspan="7">{{ r.name }}：{{ r.warnings.join('；') }}</td>
          </tr>
          <tr v-if="r.ok && r.derived_alt">
            <td :class="$style.alt" colspan="7">
              {{ r.name }}：另一套口径（{{ altName(r.perf_variant) }}）Q =
              {{ fmt(r.derived_alt.Q, 1) }} cal/g，D = {{ fmt(r.derived_alt.D, 3) }} km/s，p =
              {{ fmt(r.derived_alt.p, 2) }} GPa
            </td>
          </tr>
        </template>
      </tbody>
    </table>
  </div>

  <div :class="$style.log">{{ refs.log || '（暂无输出）' }}</div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElButton, ElSelect, ElOption, ElSwitch } from 'element-plus'
import { Refresh, VideoPlay } from '@element-plus/icons-vue'
import PanelGroup from '@/comps/Panel-Group.vue'
import PanelItem from '@/comps/Panel-Item.vue'
import type { BoomPerfVariant, BoomPredictOptions, BoomResult, wmapi_boom } from './plugin-api'
import { boomRuntime, loadBoomRuntime } from './boom/runtime'

const cores = window.wmapi_cores
const files = window.wmapi_files
const wmscene = window.wmapi_scene

const DEVICE_FOLD = 'windows / android'
const refs = ref({
  moleName: '',
  source: 'select' as 'select' | 'all',
  bondMode: 'auto' as 'auto' | 'host' | 'distance',
  indexMode: 'legacy' as 'legacy' | 'corrected',
  useDensity: true,
  useEof: true,
  derived: true,
  perfVariant: 'linear14' as BoomPerfVariant,
  busy: false,
  checking: false,
  engine: '未加载',
  engineDetail: DEVICE_FOLD,
  log: '',
  results: [] as BoomResult[],
})

const fmt = (v: number | undefined | null, n: number) => (v == null ? '—' : v.toFixed(n))
const altName = (v: BoomPerfVariant | undefined) =>
  v === 'linear14' ? 'kamlet' : 'linear14'

const targets = (): ('density' | 'eof')[] => {
  const out: ('density' | 'eof')[] = []
  if (refs.value.useDensity) out.push('density')
  if (refs.value.useEof) out.push('eof')
  return out
}

const refresh_current = () => {
  const cur = wmscene.get_mole_name() || files.get_select() || ''
  refs.value.moleName = cur
}

/** 选中的分子名列表：多选优先，其次选中，最后用场景里当前显示的分子 */
const pick_moles = (): string[] => {
  if (refs.value.source === 'all') return files.get_file_list()
  const opt = files.get_option()
  if (opt?.length) return opt
  const sel = files.get_select()
  if (sel) return [sel]
  const cur = wmscene.get_mole_name()
  return cur ? [cur] : []
}

/** 确保推理核已加载（幂等），返回自检信息 */
const ensure_engine = async () => {
  const rt = await loadBoomRuntime((msg) => {
    refs.value.log = msg
  })
  const i = rt.info
  refs.value.engine = `就绪（${i.hidden} 隐层 × ${i.pass_layers} 层，${(
    (i.params_per_model ?? 0) / 1e6
  ).toFixed(2)} M 参数/模型）`
  refs.value.engineDetail =
    `${DEVICE_FOLD}｜wasm ${(rt.sizes.wasm / 1024).toFixed(0)} KB + 权重 ` +
    `${(rt.sizes.weights / 1e6).toFixed(1)} MB｜首次加载 ${rt.loadMs.toFixed(0)} ms`
  refs.value.log =
    `推理核：Rust→wasm（进程内，无外部程序）\n` +
    `密度模型：epoch ${i.density?.epoch}，val MAE ${i.density?.val_mae?.toFixed(4)}\n` +
    `生成焓模型：epoch ${i.eof?.epoch}，val MAE ${i.eof?.val_mae?.toFixed(4)}\n` +
    `加载耗时 ${rt.loadMs.toFixed(0)} ms`
  return rt
}

/** 面板上的推理选项（全部有值，供 predict 直接使用） */
interface PredictOptions {
  targets: ('density' | 'eof')[]
  bond_mode: 'auto' | 'host' | 'distance'
  index_mode: 'legacy' | 'corrected'
  derived: boolean
  perf_variant: BoomPerfVariant
}

/** 跑一次推理：全在进程内（wasm），不再起 python 进程、不再读写结果文件。
 *
 * 模型层面的失败（元素不支持、坐标不匹配等）以 `ok:false` 的结果返回，
 * 由调用方决定怎么展示；只有「拿不到分子结构 / 推理核加载失败」才抛异常。
 */
const predict_one = async (name: string, options: PredictOptions): Promise<BoomResult> => {
  const info = await files.get_mole_info(name)
  if (!info?.syms?.length) throw new Error(`读不到 ${name} 的结构`)
  const rt = await ensure_engine()
  return rt.predict({
    token: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    name,
    syms: info.syms,
    xyzs: info.xyzs,
    bonds: info.bonds ?? null,
    bond_mode: options.bond_mode,
    index_mode: options.index_mode,
    targets: options.targets,
    derived: options.derived,
    perf_variant: options.perf_variant,
  })
}

const current_options = (): PredictOptions => ({
  targets: targets(),
  bond_mode: refs.value.bondMode,
  index_mode: refs.value.indexMode,
  derived: refs.value.derived,
  perf_variant: refs.value.perfVariant,
})

/** 面板当前设置为默认值，调用方传入的字段优先 */
const merge_options = (o?: BoomPredictOptions): PredictOptions => {
  const c = current_options()
  return {
    targets: o?.targets ?? c.targets,
    bond_mode: o?.bond_mode ?? c.bond_mode,
    index_mode: o?.index_mode ?? c.index_mode,
    derived: o?.derived ?? c.derived,
    perf_variant: o?.perf_variant ?? c.perf_variant,
  }
}

const check_env = async () => {
  refs.value.checking = true
  refs.value.log = ''
  cores.show_loading('boom 加载推理核...')
  try {
    await ensure_engine()
    cores.notify('boom 推理核就绪（Rust→wasm）', 'success')
  } catch (err) {
    refs.value.engine = '加载失败'
    refs.value.log = `${(err as Error).message}\n\n推理核与权重随插件包发布，若提示 404/缺失，请重新安装或检查安装包是否完整。`
    cores.notify('boom 推理核加载失败，请看日志', 'error')
  } finally {
    refs.value.checking = false
    cores.hide_loading()
  }
}

const run = async () => {
  const names = pick_moles()
  if (!names.length) {
    cores.notify('没有可预测的分子（先在文件列表里选中或显示一个分子）', 'warning')
    return
  }
  const tgts = targets()
  if (!tgts.length) {
    cores.notify('至少打开一个模型（密度 / 生成焓）', 'warning')
    return
  }
  refs.value.busy = true
  refs.value.results = []
  try {
    // 先加载推理核：失败就在这里给出明确原因，不必等第一个分子
    await ensure_engine()
    for (let k = 0; k < names.length; k++) {
      const name = names[k]!
      cores.show_loading(`boom 预测中 ${k + 1}/${names.length}：${name}`)
      try {
        const t0 = performance.now()
        const res = await predict_one(name, current_options())
        const ms = performance.now() - t0
        refs.value.results.push(res)
        if (res.ok) {
          const parts: string[] = []
          if (res.results.density != null) parts.push(`ρ=${res.results.density.toFixed(4)} g/cm³`)
          if (res.results.eof != null) parts.push(`ΔHf=${res.results.eof.toFixed(2)} kcal/mol`)
          if (res.derived) parts.push(`D=${res.derived.D.toFixed(3)} km/s`)
          cores.add_reslog('boom', `${name} ${parts.join('  ')}  (${ms.toFixed(0)} ms)`)
        } else {
          cores.notify(`${name} 预测失败：${res.error}`, 'error')
        }
      } catch (err) {
        refs.value.results.push({
          ok: false, name, formula: '', n_atoms: 0, n_heavy: 0, bond_mode: '', n_bonds: 0,
          results: {}, derived: null, warnings: [], error: (err as Error).message,
        })
        cores.notify(`${name} 预测异常：${(err as Error).message}`, 'error')
      }
    }
  } catch (err) {
    cores.notify(`boom 推理核不可用：${(err as Error).message}`, 'error')
    refs.value.log = `${(err as Error).message}`
  } finally {
    refs.value.busy = false
    cores.hide_loading()
  }
}

// 对外 API：脚本/其它插件可以 await window.wmapi_boom.predict('mol.xyz')
window.wmapi_boom = {
  predict: async (name, options) => {
    try {
      const res = await predict_one(name, merge_options(options))
      return [res.ok, JSON.stringify(res)] as [boolean, string]
    } catch (err) {
      return [false, (err as Error).message] as [boolean, string]
    }
  },
  predict_result: async (name, options) => predict_one(name, merge_options(options)),
  load: async () => JSON.stringify((await ensure_engine()).info),
} satisfies wmapi_boom

onMounted(() => {
  refresh_current()
  files.on_select(() => refresh_current())
  files.on_option(() => refresh_current())
  // 面板一打开就把推理核拉起来（30 MB，后台加载；失败只写日志不弹窗）
  ensure_engine().catch((err: Error) => {
    refs.value.engine = '未加载'
    refs.value.log = `预加载失败（第一次预测时会自动重试）：${err.message}`
  })
})
</script>
