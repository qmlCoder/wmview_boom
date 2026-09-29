# wmview.boom

wmview 主程序的**外部插件**：把 `模型/` 里那个 PyTorch 研究项目（GRMP 图神经网络）
变成可在 wmview 里直接调用的能力 —— 读当前/选中的分子，预测**密度 ρ** 与
**固体生成焓 ΔHf**，并给出爆热/爆速/爆压的量级估算。

推理跑在**插件自带的 Rust→wasm 推理核**里（进程内，不依赖任何外部程序、不联网），
所以 **Windows 与 Android 都可用**。

> 模型原始信息、架构、原始项目的问题清单见 [模型/README.md](模型/README.md)。
> 本文件只讲**插件侧**：怎么跑、怎么验、怎么发。

## 1. 目录结构

```
wmview.boom/
├─ src/                     # 前端（Vue3 + Module Federation remote）
│  ├─ index.ts              # 插件入口（exposes './index'）
│  ├─ Plugin.vue            # 侧边栏面板：选分子 → 跑模型 → 展示结果
│  ├─ boom/runtime.ts       # 加载 wasm + 扁平权重 + 固化表，封装 predict
│  ├─ plugin-api.d.ts       # 本插件对外 API 声明（window.wmapi_boom）
│  ├─ wmapi.d.ts            # 宿主 API 类型（从主程序同步）
│  ├─ comps/                # Panel-Group.vue / Panel-Item.vue（从主程序 src/comps 复制）
│  └─ icon.png              # 插件图标（构建时内联成 data URL）
├─ boom-core/               # ★ Rust 推理核（2644 行，编出来 284 KB wasm）
│  ├─ src/weights.rs        #   扁平权重表（索引 json + f32 bin）
│  ├─ src/table.rs          #   49 维元素描述子表
│  ├─ src/graph.rs          #   建键 / edge_index / Hill 式
│  ├─ src/geometry.rs       #   三体四体索引、距离/键角/二面角
│  ├─ src/basis.rs          #   Bessel / 球谐基组（= runtime/basis_gen.py 的 Rust 版）
│  ├─ src/model.rs          #   GRMP 前向（含手写 wasm SIMD 点积）
│  ├─ src/perf.rs           #   Q/D/p 两套实现
│  ├─ src/pipeline.rs       #   分子 → 结果 JSON（= boom_infer.py 的 run/report）
│  ├─ src/wasm.rs           #   wasm-bindgen 导出
│  ├─ src/main.rs           #   原生 CLI：--golden（对拍）/ --check / --bench
│  ├─ build-wasm.mjs        #   wasm-pack 构建（显式打开 simd128）
│  └─ verify-wasm.mjs       #   在 node 里走一遍 wasm 链路并与参考值对拍
├─ boom-data/               # ★ 随包资源：扁平权重（f32）+ 元素表 + Q/D/p 系数
├─ tools/                   # 开发期 Python 工具（不进包）
├─ runtime/                 # 迁移期的 numpy 参照实现（不进包，只用于交叉验证）
├─ 模型/                     # 原始 PyTorch 项目快照（开发侧，不进包）
├─ dist/                    # 前端构建产物（含 assets/ 里的 wasm 与权重）
├─ pack.py                  # 打包 wmview.boom.zip
└─ notes/latest.md          # 修改记录
```

## 2. 运行时方案：Rust → WASM

原模型是 PyTorch + torch_geometric + torch_scatter/sparse/cluster + rdkit + mendeleev 的栈，
既不能进 WebView，整包搬进来也有几百 MB 且 C++ 扩展 wheel 极易炸。

**第一版**把它重写成 numpy + 自带嵌入式 python 3.10（Windows-only，包 ~48 MB）。
**现在**整个推理核改成了 Rust→wasm（与同仓库的 `wmvie.flavor` 同一套做法）：

| 关注点 | python 版（旧） | Rust→wasm（现在） |
| --- | --- | --- |
| 推理位置 | 外部 `pythonw.exe` 进程 | **插件进程内**（wasm） |
| 平台 | 仅 Windows | **Windows + Android** |
| 每次预测 | 起进程 + 重读 30 MB 权重 | 权重常驻内存，直接前向 |
| 单分子 | ~300 ms 起进程 + ~100 ms 前向 | 加载后 **1~23 ms**（小分子）/ 82 ms（17 原子） |
| 观察窗 | `result.json` + token 防残留 | 直接函数返回，无文件、无竞态 |
| 包体积 | 48.4 MB | **27.9 MB**（−42%） |

移植后**随包不再有任何 Python**；`python-3.10/` 与 `tools/prepare_python.py` 已删除。

### 数值口径（必须照抄，否则对拍会从 1e-6 掉到 1e-3）

| 部分 | 精度 | 原因 |
| --- | --- | --- |
| 距离 / 键角 / 二面角 | **f64** 算完再 cast f32 | 与 `runtime/geometry.py` 的 `.astype(DT)` 一致 |
| Bessel / 球谐基组 | **f64**（含 envelope、`1/x`） | 原实现就是 f64 算完再 cast，降精度会毁掉对拍 |
| 网络前向 | f32，点积用 f32 累加 | numpy 侧同样是 f32/BLAS 累加 |
| 全局半径图的距离 | 直接用 f64（**没有** f32 往返） | 局部图是 f32 存储，两条路径精度不同，都要复现 |

矩阵乘占前向九成以上算力：标量 wasm 比原生慢约 10 倍，而 LLVM 对归约循环**不会**
自动向量化（实测开了 `+simd128` 也只快 15%），所以 `model.rs` 里点积是**手写 wasm SIMD**
（两条独立累加链），实测 204 ms → 82 ms。

### 权重格式

`tools/export_flat_weights.py` 把 `runtime/weights/*.npz` 转成
`boom-data/<target>.json`（索引）+ `<target>.bin`（f32 小端）：
npz 是 zip 容器，wasm 里解压不划算，扁平格式一次 fetch 就能当 f32 切片用。
两个模型各 226 个张量 / 3 805 320 参数 / 15.22 MB（fp16 版本 7.61 MB，已导出但**未启用**：
10 bit 尾数对 1e-4 的容差有风险，要用必须先过对拍）。

## 3. 快速上手

```bash
# 前端联调（宿主 dev 模式连 localhost:3001；也可直接装到宿主 plugs/ 下）
npm install
npm run dev

# 构建（type-check → wasm-pack 出 boom-core/pkg → vite 出 dist）
npm run build

# 打包
python pack.py          # → wmview.boom.zip（解压到主程序 plugs/wmview.boom/ 重启即加载）
```

需要 **cargo + wasm-pack**（`wasm32-unknown-unknown` target）。客户端不需要，包里只有产物。

验证（都不需要 torch；对拍门禁见 §5）：

```bash
cd boom-core && cargo build --release
./target/release/boom-core.exe --data ../boom-data --golden ../tools/golden.json  # 门禁
./target/release/boom-core.exe --data ../boom-data --check                        # 自检
./target/release/boom-core.exe --data ../boom-data --bench 200                    # 耗时
cd .. && node boom-core/verify-wasm.mjs                                           # wasm 链路
```

开发期 Python 工具（用装了 numpy 的解释器；`gen_basis`/`export_weights`/`verify_torch`
还需要 sympy/torch/rdkit/mendeleev，见 §4）：

```bash
python tools/gen_basis.py             # 重新生成 runtime/basis_gen.py
python tools/check_basis_parity.py    # 门禁：basis.rs 的常量必须与生成物一致
python tools/freeze_element_table.py  # 重新固化 49 维元素表
python tools/export_weights.py        # 模型/model_save/*.pt → runtime/weights/*.npz
python tools/export_flat_weights.py   # .npz → boom-data/*.bin|json（wasm 用）
python tools/fit_perf.py              # 重新拟合 Q/D/p 两套系数
python tools/make_icon.py             # 重新生成 src/icon.png

python tools/verify_torch.py 8        # 原 PyTorch 前向 vs numpy 实现（顺带刷新 golden.json）
python tools/check_golden.py          # numpy 参照的回归自检
python tools/compare_impls.py         # numpy 参照 vs Rust 推理核，逐字段交叉核对
```

## 4. 移植过程中的关键发现（和原始源码不一致的地方）

这几条都靠"能不能 strict=True 加载 checkpoint + 输出能不能对上"验证过：

1. **checkpoint 与 `模型/model/mix.py` 的源码不是同一版**：权重里
   `MLP.{0,2,4,6,8}.weight` 的形状是 `384→1024→512→128→64→1`，而源码写的是
   `384→512→256→128→64→1`；`adaptive_layers.*` 里存的是 `gate_net.*`（即
   `adaptive_type='gate'`），不是源码里的 `learned`。按 checkpoint 的真实结构重建后
   `load_state_dict(strict=True)` 一次通过。
   因此两个实现都**按权重文件里的形状/存在性驱动**（`Weights::child_indices`），
   换 checkpoint 只要模块结构不变就能直接用。
2. **`res_x` 是"整个层的输入"**：LocalMessagePassing / GlobalMessagePassing 里的
   残差直连用的是 `mlp_x1` **之前**的 x，不是 `mlp_x2` 的输出。这一条搞错时输出会差
   几个百分点（最初就是这样对不上的）。
3. **元素描述子的 off-by-one 必须保留**：原代码 `n = number - 1` 去索引 8 元素表，
   于是 C 用 N 的行、N 用 O 的行、O 用 F 的行。权重就是在这套特征上训练的，
   默认 `index_mode=legacy` 原样复现；`corrected` 只作对照。
4. **基础基组用 float32 零点**：原实现 `Jn_zeros` 把零点存成 float32 再代入符号表达式，
   生成器里照做，保证基组一致。
5. **`rbf.freq` 是训练过的参数**，不是 `arange(1,7)*π`，必须从权重里读（实测
   `[3.0002, 6.3427, 9.4477, ...]`）。
6. **`lin_rbf` 被同一个线性层连用了两次**（`lin_rbf(rbf_feat)` 而不是 `lin_rbf_out`），
   Rust 侧照抄。

## 5. 验证结果

正门禁是**与原 PyTorch 参考输出对拍**（`tools/golden.json` = 8 个 RDKit/ETKDG 分子 +
torch 输出，阈值 1e-4）：

| 验证 | 命令 | 结果 |
| --- | --- | --- |
| **Rust 原生 对拍 torch** | `boom-core --golden ../tools/golden.json` | **最大相对偏差 1.818e-6**（16 项） |
| **Rust wasm 对拍 torch** | `node boom-core/verify-wasm.mjs` | **最大相对偏差 2.119e-6** |
| numpy 参照 对拍 torch | `python tools/check_golden.py` | 最大相对偏差 2.14e-6 |
| Rust 对拍 numpy 参照 | `python tools/compare_impls.py` | 数值 ≤7.9e-7；**formula / 原子数 / 键数 / bond_mode / 警告文本逐字一致** |
| 基组常量 | `python tools/check_basis_parity.py` | 62/62 长常量与生成物**完全相同** |
| 结构 | 两个 checkpoint `load_state_dict(strict=True)` | 226 个张量全部对上 |
| **宿主端到端** | 装进 `plugs/` 启动主程序，经 HTTP `run_script` 调 `wmapi_boom.predict_result` | **8/8 分子与原生 CLI 一致**（ρ 差 ≤3.2e-8，ΔHf 差 0，p 差 ≤9.7e-7）；不支持的元素（U/Cr）给出干净失败信息，不崩 |
| 性能（原生 / wasm） | `--bench 200` / `verify-wasm.mjs` | 两模型+Q/D/p：**29.2 ms / 82.3 ms**（17 原子分子）；宿主内实测小分子 1~23 ms |
| 加载 | `verify-wasm.mjs` | 实例化 + 30 MB 权重：**43 ms**（node；浏览器里取决于磁盘与 asset 协议） |

## 6. Q / D / p 的两套实现

模型只输出 ρ 与 ΔHf，爆热 Q / 爆速 D / 爆压 p **不是网络输出**（见 模型/README.md §5），
是 `tools/fit_perf.py` 在随包数据集上拟合的经验关系。推理核里同时实现了两套，可由
面板的「Q/D/p 口径」或 API 的 `perf_variant` 切换，结果里的另一套放在 `derived_alt`：

| 口径 | 做法 | 留出集（20%，seed 641）中位相对误差 |
| --- | --- | --- |
| `linear14`（默认，现行） | 14 特征（元素计数 + ρ + ΔHf + 氧平衡产物）线性回归 | Q **1.070%** / D **4.194%** / p **5.348%** |
| `kamlet_noco` | Q 同左；D/p 换成**无 CO** 氧平衡 + 对数-对数 Kamlet 形式 | D **2.313%** / p **4.624%** |

`kamlet_noco` 的系数（`-0.62 / 0.517 / 1.506`）与 模型/README.md §5 记录的那组
（`-0.5735 / 0.4666 / 1.3877`）吻合 —— 说明 §5 用的是**无 CO** 规则，而
`runtime/perf.py` 实现的是**含 CO** 规则；两处口径不一致这个问题在这里一并留了记录。
UI 里两套都会显示，标注为「经验估算，非模型输出」。

## 7. 已知局限

- **适用域很窄**：训练集是 CHNO 小分子（C≤7、H≤8、N≤5、O≤4、重原子 8~10、总原子 8~17、
  ρ∈[1.2167,1.957]），超出范围时插件会在结果里给 warning，但不会阻止计算 —— 外推结果不要当真。
  含非 CHNO 元素（U/Cr 等）会直接返回 `ok:false`。
- **legacy 索引下 F 会报错**（`Z=9 → row 8` 越界），B/Li/Be 会静默取到错行 —— 这是刻意
  复现原实现，不是 bug。
- **键图与原训练不同源**：原训练用 RDKit 的 SMILES 键图；插件用宿主的键感知或距离判据。
  含氢结构影响很大（原模型是在 AddHs 后的结构上训练的），没有显式氢时会告警。
- **Q / D / p 不是模型输出**（见 §6）。
- 需要宿主 WebView 支持 **wasm SIMD**（Chromium 91+；WebView2 与 Android System WebView 都满足）。
- 权重里的 `layer_att.*` 是原项目遗留（存了但前向里从不调用），两个实现都忽略。

## 8. 插件契约

- MF remote：`name = 'wmview_boom'`，入口 `index.js`，`exposes['./index'] → src/index.ts`；
  vue / element-plus / three 由宿主 singleton 提供。
- **`vite.config.ts` 必须保留 `base: './'`**：wasm 与 29 MB 权重是 `?url` 外部资源，
  绝对的 `/assets/...` 在宿主里会解析到主机根目录而 404。
- 宿主 API：`wmapi_files`（get_mole_info / get_file_list / get_option / get_select /
  get_plugFold / wait_root …）、`wmapi_scene`（get_mole_name）、`wmapi_cores`
  （notify / add_reslog / show_loading …）。类型见 `src/wmapi.d.ts`。
  插件**不再使用** `run_exe` / `read_text`（推理已在进程内）。
- 对外 API：`window.wmapi_boom.predict(name, options)` / `predict_result(name, options)` /
  `load()`，声明在 `src/plugin-api.d.ts` 并通过 `plugin.apidoc` 导出。
- 安装包：`dist/index.js` + `dist/assets`（含 wasm 与权重）+ `info.json` + `icon.png`，
  zip 内不带顶层目录。

## 9. 发布

```bash
npm run build && python pack.py     # → wmview.boom.zip
# 上传 OSS + 入库（在 plugins/ 目录下执行，详见 ../AGENTS.md）
& $PY update.py boom --yes
```

- 版本号单一来源是 `package.json` 的 `version`，`pack.py` 会同步到 `info.json`；
  文件名固定 `wmview.boom.zip`，不带版本号。
- 只改代码不升版本号，已安装该版本的客户端不会收到更新。
- `info.json` 的 `platform` 是 `windows,android`（推理在进程内，安卓端可装）。
