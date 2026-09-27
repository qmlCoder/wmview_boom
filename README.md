# wmview.boom

wmview 主程序的**外部插件**：把 `模型/` 里那个 PyTorch 研究项目（GRMP 图神经网络）
变成可在 wmview 里直接调用的能力 —— 读当前/选中的分子，预测**密度 ρ** 与
**固体生成焓 ΔHf**，并给出爆热/爆速/爆压的量级估算。

> 模型原始信息、架构、原始项目的问题清单见 [模型/README.md](模型/README.md)。
> 本文件只讲**插件侧**：怎么跑、怎么验、怎么发。

## 1. 目录结构

```
wmview.boom/
├─ src/                     # 前端（Vue3 + Module Federation remote）
│  ├─ index.ts              # 插件入口（exposes './index'）
│  ├─ Plugin.vue            # 侧边栏面板：选分子 → 跑模型 → 展示结果
│  ├─ plugin-api.d.ts       # 本插件对外 API 声明（window.wmapi_boom），经 plugin.apidoc 导出
│  ├─ wmapi.d.ts            # 宿主 API 类型（从主程序同步）
│  ├─ comps/                # Panel-Group.vue / Panel-Item.vue（从主程序 src/comps 复制）
│  └─ icon.png              # 插件图标（构建时内联成 data URL）
├─ runtime/                 # 推理运行时（随包分发，只依赖 numpy）
│  ├─ boom_infer.py         # 入口：stdin 读分子 → 跑模型 → 写 out/result.json
│  ├─ grmp_numpy.py         # GRMP 前向的 numpy 实现（对齐 模型/model/mix.py + layer/geom.py）
│  ├─ geometry.py           # 键图几何特征（距离/键角/二面角、三体四体索引）
│  ├─ basis_gen.py          # 自动生成：球贝塞尔/球谐基组（tools/gen_basis.py）
│  ├─ features.py           # 元素 → 49 维描述子、Hill 式、距离判据建键
│  ├─ perf.py               # Q / D / p 经验估算
│  └─ weights/              # density.npz / eof.npz（由 .pt 转换）+ 固化表 + 系数
├─ tools/                   # 开发期工具（不进包）
├─ 模型/                     # 原始 PyTorch 项目快照（开发侧，不进包）
├─ python-3.10/             # 内置 python 运行时（tools/prepare_python.py 准备，不进 git）
├─ pack.py                  # 打包 wmview.boom.zip
└─ notes/latest.md          # 修改记录
```

## 2. 运行时方案：为什么不是 torch

模型是 PyTorch + torch_geometric + torch_scatter/sparse/cluster + rdkit + mendeleev 的栈，
既不能进 WebView，整包搬进来也有几百 MB 且 C++ 扩展 wheel 极易炸。

所以做法是**把推理重写成 numpy**（网络里只有 MLP / sin / Bessel / 球谐 / scatter_add /
LayerNorm，没有不可移植算子）：

| 原始依赖 | 插件里的替代 | 说明 |
| --- | --- | --- |
| torch + checkpoint | `runtime/weights/*.npz` | export_weights.py 只做 state_dict → numpy fp32，不改数值 |
| torch_geometric / scatter / sparse / cluster | `runtime/geometry.py` | 键图几何特征、半径图、scatter_add 的等价实现 |
| sympy 现场推导的基组 | `runtime/basis_gen.py` | gen_basis.py 用 sympy 推出表达式后固化成 numpy 代码 |
| mendeleev + sklearn 现算的 49 维描述子 | `runtime/weights/element_table.json` | 固化成常量表，避免上游数据更新导致列序静默错位 |
| rdkit 键图 | 宿主 `wmapi_files.get_mole_info().bonds`，没有时用共价半径距离判据 | 默认 `bond_mode=auto` |

运行时只有 numpy 一个第三方依赖，随插件带一份嵌入式 python 3.10（约 75MB 展开 / 50MB 压缩包）。

## 3. 快速上手

```bash
# 前端（主程序开发模式会自动连 localhost:3001）
npm install
npm run dev

# 准备内置 python 运行时（拷 ../python-3.10 + 装 numpy + 精简）
python tools/prepare_python.py

# 环境自检（不需要 torch）
./python-3.10/python.exe runtime/boom_infer.py --check

# 构建 + 打包
npm run build
python pack.py          # → wmview.boom.zip（解压到主程序 plugs/wmview.boom/ 重启即加载）
```

开发期工具（需要 miniconda 的 python，装了 torch/rdkit/mendeleev/sympy）：

```bash
python tools/gen_basis.py             # 重新生成 runtime/basis_gen.py
python tools/freeze_element_table.py  # 重新固化 49 维元素表
python tools/export_weights.py        # 模型/model_save/*.pt → runtime/weights/*.npz
python tools/fit_perf.py              # 重新拟合 Q/D/p 经验系数
python tools/make_icon.py             # 重新生成 src/icon.png

# 数值对齐验证：用原始 PyTorch 代码跑前向，与 numpy 实现逐分子比对
python tools/verify_torch.py 8        # 顺带刷新 tools/golden.json
python tools/check_golden.py          # 不需要 torch 的回归自检
```

## 4. 移植过程中的关键发现（和原始源码不一致的地方）

这几条都靠"能不能 strict=True 加载 checkpoint + 输出能不能对上"验证过：

1. **checkpoint 与 `模型/model/mix.py` 的源码不是同一版**：权重里
   `MLP.{0,2,4,6,8}.weight` 的形状是 `384→1024→512→128→64→1`，而源码写的是
   `384→512→256→128→64→1`；`adaptive_layers.*` 里存的是 `gate_net.*`（即
   `adaptive_type='gate'`），不是源码里的 `learned`。按 checkpoint 的真实结构重建后
   `load_state_dict(strict=True)` 一次通过。
   因此本插件的 numpy 实现**按权重文件里的形状/存在性驱动**（层数、MLP 宽度、融合方式都从
   npz 读），换 checkpoint 只要模块结构不变就能直接用。
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

## 5. 验证结果

| 验证 | 内容 | 结果 |
| --- | --- | --- |
| 结构 | 原 GRMP + 按 checkpoint 修正 MLP 后 `load_state_dict(strict=True)` | 通过（226 个张量全部对上） |
| 数值 | `tools/verify_torch.py`：原 PyTorch 前向 vs numpy 前向，8 个 RDKit/ETKDG 分子 × 2 个模型 | 最大相对偏差 **1.6e-6**（float32 噪声级） |
| 回归 | `tools/check_golden.py`（不依赖 torch，用内置 python） | 通过 |
| 宿主联调 | 经主程序 HTTP `run_script` 调 `wmapi_cores.run_exe` 跑内置 python + `read_text` 读回结果 | 通过（token、ρ、ΔHf、Q/D/p 全部正确） |

## 6. 已知局限

- **只在 Windows 上可用**（依赖内置 python 运行时）。
- **适用域很窄**：训练集是 CHNO 小分子（C≤7、H≤8、N≤5、O≤4、原子数 ≤17、ρ∈[1.2,2.0]），
  超出范围时插件会在结果里给 warning，但不会阻止计算 —— 外推结果不要当真。
- **键图与原实现不同源**：原训练用 RDKit 的 SMILES 键图；插件用宿主的键感知或距离判据。
  含氢结构影响很大（原模型是在 AddHs 后的结构上训练的），没有显式氢时会告警。
- **Q / D / p 不是模型输出**：是 `tools/fit_perf.py` 在随包 CSV 上的线性回归
  （留出集：Q 中位误差 1.1%、D 4.2%、p 5.4%），UI 里也标注为"经验估算"。
- 权重里的 `layer_att.*` 是原项目遗留（存了但前向里从不调用），实现里也一并忽略。

## 7. 插件契约

- MF remote：`name = 'wmview_boom'`，入口 `index.js`，`exposes['./index'] → src/index.ts`；
  vue / element-plus / three 由宿主 singleton 提供。
- 宿主 API：`wmapi_cores`（run_exe / read_text / notify / add_reslog / show_loading …）、
  `wmapi_files`（get_mole_info / get_option / get_select / get_file_list / get_plugFold …）、
  `wmapi_scene`（get_mole_name）。类型见 `src/wmapi.d.ts`。
- `run_exe(source, exe, args, input)`：`source` 固定 `plugin_boom`；stdout 由宿主按 GBK 解码；
  结果以文件（`runtime/out/result.json`）为准，并用 token 防止读到上一次的残留。
- 对外 API：`window.wmapi_boom.predict(name)` / `predict_result(name)`，声明在
  `src/plugin-api.d.ts` 并通过 `plugin.apidoc` 导出，脚本与其它插件可直接调用。

## 8. 发布

```bash
npm run build && python pack.py     # → wmview.boom.zip
# 上传 OSS + 入库（在 plugins/ 目录下执行，详见 ../AGENTS.md）
& $PY update.py boom --yes
```

- 版本号单一来源是 `package.json` 的 `version`，`pack.py` 会同步到 `info.json`；
  文件名固定 `wmview.boom.zip`，不带版本号。
- 只改代码不升版本号，已安装该版本的客户端不会收到更新。
