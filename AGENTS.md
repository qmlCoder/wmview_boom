# wmview.boom 插件项目规则

wmview 主程序的**外部插件**：把 `模型/` 里的 GRMP 图神经网络搬到插件里，
从分子结构预测密度与固体生成焓（含 Q/D/p 经验估算）。

推理核是**插件自带的 Rust→wasm**（进程内，无外部程序），Windows 与 Android 都可用。

> 本文件是本项目的核心规则，开工前先读；与任务冲突时以本文件为准并向用户说明。

## 参考项目（必读）

| 项目 | 路径（相对本目录） | 用途 |
| --- | --- | --- |
| 主程序 | `../../wmview_app` | API、库版本、插件契约的**唯一权威来源** |
| WASM 插件样板 | `../wmvie.flavor`（`flavor-core/`） | **本插件推理核的写法来源**：Cargo 结构、扁平权重、`--parity`/`verify-wasm.mjs`、`base: './'` + `?url` |
| 调外部程序的样板 | `../wmview.xtb`、`../wmview.xyzrender` | 只在需要外部进程时参考（本插件已不用） |
| 上层规则 | `../AGENTS.md` | 插件清单、发布链路（OSS / 数据库 / update.py） |

## 依赖与版本（硬约束）

- 运行时库复用主程序 MF 共享依赖：`vue` / `element-plus` / `three` 由主程序（MF Host）
  以 singleton 提供，插件不重复打包；`@module-federation/vite` 版本与主项目保持一致。
- 前端**禁止**引入主项目没有的运行时依赖。确有必要时先提主项目修改意见，不要在本项目内硬绕。
- **随包运行时不带 Python**：推理全在 `boom-core`（Rust→wasm）里。Rust 侧只允许
  `serde` / `serde_json`（+ wasm32 下的 `wasm-bindgen`），新增 crate 前必须说明理由。
- 构建需要 `cargo` + `wasm-pack`（`wasm32-unknown-unknown`）；`wasm-pack` 会覆盖
  `RUSTFLAGS`，所以 **SIMD 只能由 `boom-core/build-wasm.mjs` 显式打开**，
  不要指望 `.cargo/config.toml` 里的 `[target.*] rustflags`（实测被忽略）。
- `tools/` 下的 Python 工具（`gen_basis` / `export_weights` / `verify_torch` 等）
  是**开发侧**的，需要 numpy/sympy/torch/rdkit/mendeleev，**不进包**，不要试图去 Python 化。

## 插件契约（勿擅自更改）

- MF remote：`name` 必须是 `wmview_boom`（主程序 `remoteNameOf('wmview.boom')`），
  入口文件名固定 `index.js`，`exposes['./index']` 指向 `src/index.ts`。
- `src/index.ts` 必须同时导出 `plugin`（生产/本地安装）与 `default`（开发模式）。
- **`vite.config.ts` 必须保留 `base: './'`**：wasm 与 ~29 MB 权重是 `?url` 外部资源，
  `/assets/...` 绝对路径在宿主里会解析到主机根目录而 404（这是 asset 协议的解析规则，
  见主程序 `src/plugin/mf.ts` 的注释）。
- 宿主 API 一律走 `window.wmapi_*`，类型只在 `src/wmapi.d.ts` 维护（与主程序同步），
  不要用 `inject('scene')` 之类的内部通道。**不要再用 `run_exe` / `read_text`**：
  推理在进程内完成，没有外部进程、没有结果文件、没有 token 防残留。
- 对外能力挂在 `window.wmapi_boom`，声明写在 `src/plugin-api.d.ts`，并作为 `plugin.apidoc` 导出。
  模型层面的失败（元素不支持等）以 `ok:false` 返回，**不抛异常**；只有拿不到结构 /
  推理核加载失败才抛。

## 模型侧规则（重要）

- **不要"修正"原实现的怪癖**：权重是在那套特征/结构上训练的。必须保留的怪癖见
  `README.md` §4（元素索引 off-by-one、`res_x` 语义、`lin_rbf` 连用两次、
  checkpoint 与源码的 MLP 宽度不一致、`rbf.freq` 是训练参数）。
  Rust（`boom-core/src/model.rs`）与 numpy 参照（`runtime/grmp_numpy.py`）都必须照抄。
- **数值精度分工不能改**：几何与基组用 **f64** 算完再 cast f32；网络用 f32。
  详见 `README.md` §2，改错会让对拍从 1e-6 掉到 1e-3。
- 结构（层数、MLP 宽度、融合方式）必须**从权重文件读**，不要写死：
  Rust 用 `Weights::child_indices`，numpy 用 `_child_indices`。
- **`runtime/basis_gen.py` 与 `boom-core/src/basis.rs` 是同一批常量的两份实现**：
  前者是 `tools/gen_basis.py` 的产物（不要手改），后者是从它誊写的。
  改了任一侧都要跑 `python tools/check_basis_parity.py`（门禁：长常量必须完全一致）。
- **对拍门禁（改动数值实现后必须全跑）**：
  1. `cd boom-core && cargo build --release && ./target/release/boom-core.exe --data ../boom-data --golden ../tools/golden.json`
     —— 与原 PyTorch 参考输出对拍，阈值 `TOL=1e-4`（float32 前向的噪声上限，不是精度承诺）；
  2. `node boom-core/verify-wasm.mjs` —— 同一条门禁，但走完整 wasm 边界；
  3. `python tools/check_basis_parity.py` —— 基组常量一致性；
  4. 改了权重/几何/基组还要用 `python tools/verify_torch.py 8` 重新和原 PyTorch 对齐
     （需要 torch + rdkit 的 miniconda 环境），它会顺带刷新 `tools/golden.json`。
  更细的逐字段核对用 `python tools/compare_impls.py`（numpy 参照 vs Rust）。
- 换了 checkpoint：`tools/export_weights.py` → `tools/export_flat_weights.py`，然后跑门禁。

## 目录与命令

```bash
npm install && npm run dev      # 前端联调（宿主 dev 模式连 localhost:3001）
npm run build                   # type-check → wasm-pack(boom-core/pkg) → dist/index.js + dist/assets
node boom-core/build-wasm.mjs   # 只重建 wasm（含 simd128）
node boom-core/verify-wasm.mjs  # 只验 wasm 链路
python pack.py                  # → wmview.boom.zip
```

- `dist/`、`boom-core/pkg/`、`boom-core/target/`、`runtime/out/`、`*.zip` 不进版本库（见 `.gitignore`）。
- `模型/` 是原始项目快照，只作开发参考，**不进安装包**；`boom-data/`（扁平权重）**要进包**。
- `boom-data/*_f16.bin` 是 fp16 体积/精度实验产物，当前**未启用**；启用前必须先过对拍门禁。

## 版本号与发布

- 版本号单一来源是 `package.json` 的 `version`；`pack.py` 打包时同步进 `info.json`。
- 压缩包文件名固定 `wmview.boom.zip`，不带版本号。
- `info.json` 的 `platform` 为 `windows,android`（推理在进程内，安卓端可装）。
- 发布流程：改代码 → 升版本号 → `npm run build` → `python pack.py` →
  在 `plugins/` 下 `python update.py boom --yes`（详见 `../AGENTS.md`）。

## 修改记录

- 每完成一次功能/修复，把要点追加到 `notes/latest.md`
  （`- 新增:` / `- 修复:` / `- 验证:` / `- 发布:`）。
