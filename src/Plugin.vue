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
    <panel-item title="估算 Q / D / p" tip="经验拟合关系（不是模型输出）：Q 误差约 1%，D/p 约 4~5%">
      <el-switch v-model="refs.derived"></el-switch>
    </panel-item>
  </panel-group>

  <div :class="$style.rowBtns">
    <el-button :icon="Refresh" :loading="refs.checking" @click="check_env">检查环境</el-button>
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
        <tr v-for="(r, k) in refs.results" :key="k">
          <template v-if="r.ok">
            <td>{{ r.name }}</td>
            <td>{{ r.formula }}</td>
            <td>{{ fmt(r.results.density, 4) }}</td>
            <td>{{ fmt(r.results.eof, 2) }}</td>
            <td>{{ fmt(r.derived?.Q, 1) }}</td>
            <td>{{ fmt(r.derived?.D, 3) }}</td>
            <td>{{ fmt(r.derived?.p, 2) }}</td>
          </template>
          <template v-else>
            <td>{{ r.name }}</td>
            <td :class="$style.warn" colspan="6">{{ r.error }}</td>
          </template>
        </tr>
        <tr v-for="(r, k) in refs.results" :key="'w' + k">
          <td :class="$style.warn" colspan="7" v-if="r.warnings?.length">
            {{ r.name }}：{{ r.warnings.join('；') }}
          </td>
        </tr>
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
import type { BoomResult, wmapi_boom } from './plugin-api'

const cores = window.wmapi_cores
const files = window.wmapi_files
const wmscene = window.wmapi_scene

// 宿主 run_exe 的日志前缀
const SOURCE = 'plugin_boom'
// 开发时插件跑在源码目录（内置 python 就在这），发布时跑在宿主 plugs 下的安装目录
const DEV_FOLD = 'E:/code/wmview/plugins/wmview.boom'

const refs = ref({
  moleName: '',
  source: 'select' as 'select' | 'all',
  bondMode: 'auto' as 'auto' | 'host' | 'distance',
  indexMode: 'legacy' as 'legacy' | 'corrected',
  useDensity: true,
  useEof: true,
  derived: true,
  busy: false,
  checking: false,
  log: '',
  results: [] as BoomResult[],
})

const fmt = (v: number | undefined, n: number) => (v == null ? '—' : v.toFixed(n))

/** 插件目录：dev 用本地源码目录，发布用宿主 plugs 下的安装目录 */
let foldP: Promise<string> | undefined
const plugin_fold = () => {
  if (!foldP) {
    foldP = (async () => {
      if (import.meta.env.DEV) return DEV_FOLD
      await files.wait_root()
      return `${files.get_plugFold()}/wmview.boom`
    })()
  }
  return foldP
}

const py_exe = (fold: string) => `${fold}/python-3.10/pythonw.exe`
const infer_script = (fold: string) => `${fold}/runtime/boom_infer.py`
const result_file = (fold: string) => `${fold}/runtime/out/result.json`

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

/** 跑一次推理：把分子喂给 runtime/boom_infer.py，读回 runtime/out/result.json */
const predict_one = async (name: string, fold: string, tgts: ('density' | 'eof')[]): Promise<BoomResult> => {
  const info = await files.get_mole_info(name)
  if (!info?.syms?.length) throw new Error(`读不到 ${name} 的结构`)
  const token = `${Date.now()}-${Math.random().toString(36).slice(2)}`
  const payload = {
    token,
    name,
    syms: info.syms,
    xyzs: info.xyzs,
    bonds: info.bonds ?? null,
    bond_mode: refs.value.bondMode,
    index_mode: refs.value.indexMode,
    targets: tgts,
    derived: refs.value.derived,
  }
  const [ok, out] = await cores.run_exe(
    SOURCE, py_exe(fold), [infer_script(fold), '--out', result_file(fold)], JSON.stringify(payload),
  )
  if (out) refs.value.log = out
  let text = ''
  try {
    text = await cores.read_text(result_file(fold))
  } catch (err) {
    throw new Error(`读不到结果文件（是否已准备内置 python：python tools/prepare_python.py）：${(err as Error).message}`)
  }
  let res: BoomResult
  try {
    res = JSON.parse(text) as BoomResult
  } catch (err) {
    throw new Error(`结果文件不是合法 JSON：${(err as Error).message}`)
  }
  if (res.token !== token) throw new Error('结果文件不是本次运行的输出（令牌不匹配）')
  if (!ok && res.ok !== false) throw new Error('推理进程异常退出，详情见日志')
  return res
}

const check_env = async () => {
  refs.value.checking = true
  refs.value.log = ''
  cores.show_loading('boom 检查运行环境...')
  try {
    const fold = await plugin_fold()
    const [ok, out] = await cores.run_exe(SOURCE, py_exe(fold), [infer_script(fold), '--check'], '')
    refs.value.log = out || '（没有输出：内置 python 运行时可能还没准备好，请先运行 python tools/prepare_python.py）'
    cores.notify(ok ? 'boom 运行环境正常' : 'boom 运行环境检查失败，请看日志', ok ? 'success' : 'error')
  } catch (err) {
    refs.value.log = `${(err as Error).message}`
    cores.notify('boom 运行环境检查异常', 'error')
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
  refs.value.log = ''
  const fold = await plugin_fold()
  try {
    for (let k = 0; k < names.length; k++) {
      const name = names[k]!
      cores.show_loading(`boom 预测中 ${k + 1}/${names.length}：${name}`)
      try {
        const res = await predict_one(name, fold, tgts)
        refs.value.results.push(res)
        if (res.ok) {
          const parts: string[] = []
          if (res.results.density != null) parts.push(`ρ=${res.results.density.toFixed(4)} g/cm³`)
          if (res.results.eof != null) parts.push(`ΔHf=${res.results.eof.toFixed(2)} kcal/mol`)
          if (res.derived) parts.push(`D=${res.derived.D.toFixed(3)} km/s`)
          cores.add_reslog('boom', `${name} ${parts.join('  ')}`)
        } else {
          cores.notify(`${name} 预测失败：${res.error}`, 'error')
        }
      } catch (err) {
        refs.value.results.push({
          ok: false, name, formula: '', n_atoms: 0, n_heavy: 0, bond_mode: '',
          results: {}, derived: null, warnings: [], error: (err as Error).message,
        })
        cores.notify(`${name} 预测异常：${(err as Error).message}`, 'error')
      }
    }
  } finally {
    refs.value.busy = false
    cores.hide_loading()
  }
}

// 对外 API：脚本/其它插件可以 await window.wmapi_boom.predict('mol.xyz')
window.wmapi_boom = {
  predict: async (name, tgts) => {
    try {
      const res = await predict_one(name, await plugin_fold(), tgts ?? ['density', 'eof'])
      return [res.ok, JSON.stringify(res)] as [boolean, string]
    } catch (err) {
      return [false, (err as Error).message] as [boolean, string]
    }
  },
  predict_result: async (name, tgts) => predict_one(name, await plugin_fold(), tgts ?? ['density', 'eof']),
} satisfies wmapi_boom

onMounted(() => {
  refresh_current()
  files.on_select(() => refresh_current())
  files.on_option(() => refresh_current())
})
</script>
