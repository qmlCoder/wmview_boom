# wmview.boom 插件项目规则

wmview 主程序的**外部插件**：把 `模型/` 里的 GRMP 图神经网络搬到插件里，
从分子结构预测密度与固体生成焓（含 Q/D/p 经验估算）。

> 本文件是本项目的核心规则，开工前先读；与任务冲突时以本文件为准并向用户说明。

## 参考项目（必读）

| 项目 | 路径（相对本目录） | 用途 |
| --- | --- | --- |
| 主程序 | `../../wmview_app` | API、库版本、插件契约的**唯一权威来源** |
| 插件样板 | `../wmview.xtb`、`../wmview.xyzrender` | 工程结构 / 打包 / 调外部程序的写法参考 |
| 上层规则 | `../AGENTS.md` | 插件清单、发布链路（OSS / 数据库 / update.py） |

## 依赖与版本（硬约束）

- 运行时库复用主程序 MF 共享依赖：`vue` / `element-plus` / `three` 由主程序（MF Host）
  以 singleton 提供，插件不重复打包；`@module-federation/vite` 版本与主项目保持一致。
- 前端**禁止**引入主项目没有的运行时依赖。确有必要时先提主项目修改意见，不要在本项目内硬绕。
- python 侧：推理运行时**只允许依赖 numpy**（随插件带内置 python 3.10）。
  新增第三方依赖前必须说明理由，并同步更新 `tools/prepare_python.py` 与打包清单。

## 插件契约（勿擅自更改）

- MF remote：`name` 必须是 `wmview_boom`（主程序 `remoteNameOf('wmview.boom')`），
  入口文件名固定 `index.js`，`exposes['./index']` 指向 `src/index.ts`。
- `src/index.ts` 必须同时导出 `plugin`（生产/本地安装）与 `default`（开发模式）。
- 宿主 API 一律走 `window.wmapi_*`，类型只在 `src/wmapi.d.ts` 维护（与主程序同步），
  不要用 `inject('scene')` 之类的内部通道。
- `run_exe(source, exe, args, input)`：`source` 固定 `plugin_boom`；stdout 由宿主按 GBK 解码
  （python 打印中文要走 GBK，或做 `errors='replace'` 兜底）；
  **结论性数据一律写文件**（`runtime/out/result.json`）再 `read_text` 读回，
  并用 token 校验，避免读到上一次运行的残留。
- 对外能力挂在 `window.wmapi_boom`，声明写在 `src/plugin-api.d.ts`，并作为 `plugin.apidoc` 导出。

## 模型侧规则（重要）

- **不要改 `runtime/grmp_numpy.py` 去"修正"原实现的怪癖**：权重是在那套特征/结构上训练的。
  已知且必须保留的怪癖见 `README.md` §4（元素索引 off-by-one、`res_x` 语义、
  checkpoint 与源码的 MLP 宽度不一致、`rbf.freq` 是训练参数）。
- 结构（层数、MLP 宽度、融合方式）必须**从权重文件读**，不要写死：`GRMP` 类里用
  `_child_indices` 探测，保证换 checkpoint 不用改代码。
- 任何数值实现改动后，必须跑 `tools/check_golden.py`；动了几何/基组/网络结构还要用
  `tools/verify_torch.py` 重新和原 PyTorch 对齐（需要 torch + rdkit 的 miniconda 环境）。
- `runtime/basis_gen.py` 是 `tools/gen_basis.py` 的产物，**不要手改**。
- `tools/check_golden.py` 的阈值 `TOL=1e-4` 是 float32 前向的噪声上限，不是精度承诺。

## 目录与命令

```bash
npm install && npm run dev      # 前端联调（宿主自动连 localhost:3001）
npm run build                   # → dist/index.js + dist/assets
python tools/prepare_python.py  # 准备内置 python（含 numpy）
python pack.py                  # → wmview.boom.zip
```

- `dist/`、`python-3.10/`、`runtime/out/`、`*.zip` 不进版本库（见 `.gitignore`）。
- `模型/` 是原始项目快照，只作开发参考，**不进安装包**。

## 版本号与发布

- 版本号单一来源是 `package.json` 的 `version`；`pack.py` 打包时同步进 `info.json`。
- 压缩包文件名固定 `wmview.boom.zip`，不带版本号。
- 发布流程：改代码 → 升版本号 → `npm run build` → `python pack.py` →
  在 `plugins/` 下 `python update.py boom --yes`（详见 `../AGENTS.md`）。

## 修改记录

- 每完成一次功能/修复，把要点追加到 `notes/latest.md`
  （`- 新增:` / `- 修复:` / `- 验证:` / `- 发布:`）。
