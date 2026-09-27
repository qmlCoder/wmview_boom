# GRMP —— 含能材料性质预测模型（wmview.boom 内核）

> **⚠️ 2026-09-27 更正（插件侧移植时实测）：**
> 本文下面 §3/§4 有两处**看错了**，已就地改正：
> 1. 融合层不是 `'learned'` 而是 **`'gate'`**（checkpoint 里存的是 `adaptive_layers.*.gate_net.*`，
>    没有任何 `ratio` 参数）；
> 2. 最后的 MLP 不是 `384→512→256→128→64→1`，权重里的形状是 **`384→1024→512→128→64→1`**。
>    **checkpoint 与当前 `model/mix.py` 源码不是同一版**，直接 `load_state_dict` 会失败；
>    按权重实际结构重建后 `strict=True` 可通过。详见 `../README.md` §4。

> 本 README 由**通读源码 + 实测权重/数据**写成（2026-09-27），不是照抄注释。
> 所有数字都标注了来源：代码行号、从 `.pt` pickle 里解出的元数据、或对 CSV 的统计/回归。

本目录是 Linux HPC（`/export/home/zhengym`）上一个 PyTorch 研究项目的**部分快照**，
预期作为 wmview 新插件 `wmview.boom`（爆轰性能预测）的模型内核。

⚠️ **现状：这里不能训练、也不能直接推理。** 预处理产物（`*.pt` 数据集）与 xyz 结构库都没带过来，
代码里全是 HPC 绝对路径，且训练脚本在 import 阶段就会开跑。详见 [§6 问题清单](#6-问题清单)。

---

## 1. 目录结构

```
模型/
├── data/
│   ├── features.py          # 特征工程 + 建 torch_geometric 数据集（入口，模块级直接执行）
│   └── final_123679.csv     # 标签表 123 679 × 9
├── dataset/data_load.py     # 读 .pt → train/val/test 划分 → DataLoader（模块级副作用）
├── layer/geom.py            # LocalMessagePassing（键图）/ GlobalMessagePassing（半径图）
├── model/
│   ├── mix.py               # GRMP 主网络 + AdaptiveMixtureLayer（局部/全局自适应融合）
│   └── ema.py               # EMA 权重滑动平均
├── utils/
│   ├── utils.py             # MLP / ResidualLayer / glorot_orthogonal
│   └── special.py           # Bessel、球谐、二面角基组；几何索引与 距离/键角/二面角 计算
├── fea_den.py               # 训练脚本：密度      （与 fea_eof.py 除目标列外完全相同）
├── fea_eof.py               # 训练脚本：生成焓 HOF
├── model_save/
│   ├── best_density.pt      # 15.3 MB / 3 805 320 参数
│   └── best_eof.pt          # 15.3 MB / 3 805 320 参数
└── 解释.txt                  # 原作者写的 6 步导读（运行顺序）
```

`wmview.boom/` 目前**还没有** `README.md`(本文件除外)/`package.json`/`src/`/`info.json`/`pack.py`，
也不是 git 仓库，`plugins/update.py` 里没有它的登记 —— **插件本身一行代码都还没写**。
同级的 `wmview.boom/模型.zip`（32 MB，只读）与本目录内容重复，属于冗余归档，不应进版本库。

原作者的阅读顺序（`解释.txt`）：`data/features.py` → `dataset/data_load.py` → `utils/*` → `layer/geom.py` →
`model/mix.py` → `python fea_den.py`。

---

## 2. 数据：`data/final_123679.csv`

列：`number, filename, smiles, Formula, density(g/cm3), solid_HOF(kcal/mol), Q(cal/g), D(Km/s), p(GPa)`

| 项目 | 实测值 | 说明 |
| --- | --- | --- |
| 样本数 | **123 679** | train 98 945 / val 12 367 / **test 2 367**（`data_load.py:24` 的 `98945 + 12367`） |
| 元素 | **只有 C / H / N / O** | 2 万行抽样中 100% 是这四种 |
| 唯一 Formula | **仅 71 种** | 主力：C5H7N3O2、C7H7NO2、C6H6N2O2、C6H7NO3、C6H7N3O |
| 重原子数 | 8 ~ 10，总原子数均值 16.2（最大 17） | GDB 级小分子 |
| 密度 ρ | 1.217 ~ 1.957，均值 1.480，**σ = 0.083** | 取值范围极窄 |
| HOF | −124.1 ~ 140.4 kcal/mol，σ = 37.4 | |
| Q / D / p | −589.9 ~ 2109.9 cal/g；0.318 ~ 9.219 km/s；0.039 ~ 39.575 GPa | 后三列是**算出来的**，见 §5 |
| 重复 SMILES | 2 条 | 基本无重复 |

**化学空间极窄**（C5–C7、8–10 重原子的 CHNO）是本项目最重要的一条约束：
任何超出该范围的分子都属于外推，不要相信模型输出。

### 特征构成（`data/features.py`）

- **`x_geom`（模型真正使用的输入）**：每原子 **49 维** —— 用 mendeleev 取 8 个元素
  (`ele_nums = [1] + list(range(3,10))` → H, Li, Be, B, C, N, O, F) 的
  `原子序数 / 第二电离能 / 原子半径 / 周期 / 价电子数 / 原子体积 / 电子亲和能`，
  每个物理量单独做 `OneHotEncoder` 再拼接（合计 49 列，与 `mix.py:87` 的 `nn.Linear(49, …)` 对齐）。
- **`x_graph` / `edge_attr` / `geom_edge_attr` / `geom_edge_index`**：也在 `features.py` 里算了
  （RDKit 16 维 2D 原子描述子、6 维键描述子、41 维高斯距离展开），但 **`GRMP.forward` 完全没用**。
- `pos` 来自 xyz；`edge_index` 来自 **RDKit（SMILES 加 H 后）的键图** —— 几何量（距离/键角/二面角）
  是按这个键图算的，所以"拓扑"和"几何"来自两个不同的输入。

---

## 3. 模型架构（`model/mix.py`：GRMP = Global–Radial Message Passing?）

```
x_geom[49] ──Linear──► x[128]
                         │
       ┌─────────────────┴──────────────────────────────┐
       │  重复 pass_layers = 3 层                        │
       │   LocalMessagePassing  ← 键图 edge_index        │
       │     ├ rbf : BesselBasisLayer(num_radial=6)                    距离
       │     ├ sbf : SphericalBasisLayer(3×6=18)                       距离+键角
       │     └ tbf : torsionLayer(3×3×6=54)                            距离+键角+二面角
       │   GlobalMessagePassing ← radius(pos, 7.0Å) 全邻接图（41 维高斯边特征）
       │   AdaptiveMixtureLayer(local, global)   ← learned 软权重融合 + LayerNorm + Dropout0.1
       └─────────────────┬──────────────────────────────┘
                 concat 3 层节点表示 → [N, 3×128=384]
                 global_mean_pool → [B, 384]
                 MLP 384→1024→512→128→64→1   （ReLU，宽度取自 checkpoint，与源码不同）
```

- **参数量 3 805 320**（≈3.8 M，fp32 15.3 MB），输出 1 个标量。
- 融合层三选一（`adaptive_type='learned' / 'attention' / 'gate'`）；
  **从 checkpoint 的 key 反查，实际用的是 `'gate'`**：存的是 `adaptive_layers.*.gate_net.*`，
  没有 `ratio` / `attention_net`，另两条分支是死代码（此处为 2026-09-27 更正）。
- 池化只有 `global_mean_pool`（`import` 了 `global_max_pool` 但没用）。
- `self.layer_att`（128→64→1，`mix.py:110`）训练时**被保存进权重却从不调用**，是遗留的层级注意力残骸。
- 基组（`utils/special.py`）在 **import 期用 sympy 符号推导** Bessel 零点与球谐多项式再 `lambdify`，
  因此运行时依赖 sympy，且加载模型有额外开销。

---

## 4. 训练配置与真实成绩

`fea_den.py` 与 `fea_eof.py` 逐行 diff 只差：目标列（`data.M` vs `data.F`）、日志文件名、存盘文件名、
以及绘图里的单位文案 —— **两份 12 KB 脚本 95% 重复**。

| 配置 | 值 |
| --- | --- |
| epochs | 100 |
| batch_size | 64 |
| lr | 5e-4，`StepLR(step_size=15, gamma=0.5)` |
| loss | **L1Loss**（训练/验证/测试一致） |
| 裁剪 | `clip_grad_norm_(max_norm=1000)` —— 形同虚设 |
| EMA | decay 0.999，**存盘前 `ema.assign`，所以权重里已经是 EMA 参数** |
| 划分 | 随机划分（seed 641），非骨架/时间切分 |

从两个 `.pt` 的 pickle 尾部解出的真实成绩：

| checkpoint | 目标 | 存盘 epoch | val MAE | 说明 |
| --- | --- | --- | --- | --- |
| `best_density.pt` | 密度 (g/cm³) | **44** | **0.0202** | ≈ 0.24 σ，正常水平 |
| `best_eof.pt` | 固体生成焓 (kcal/mol) | **98** | **0.5802** | 相对 σ=37.4 好到不合常理 |

两点解读：

1. `best_eof` 在第 98/100 epoch 才创最好值 → **训练并未收敛就停了**，加 epoch 大概率还能再降。
2. HOF 的 0.58 kcal/mol 更像是"标签本身由同一套半经验/平滑流程生成"或"随机划分导致近似同分异构体泄漏"，
   **真实外推性能要打折看**，不能当作可对外宣称的精度。

加载示例（**必须 map_location**，checkpoint 的 storage 是 `cuda:0`）：

```python
import torch
from model.mix import GRMP

ckpt = torch.load("model_save/best_density.pt", map_location="cpu")
model = GRMP(hidden_channels=128, g_cutoff=7.0, cutoff=5.0, num_spherical=3, num_radial=6,
             envelope_exponent=5, basis_emb_size_dist=128, basis_emb_size_angle=256,
             basis_emb_size_torsion=512, emb_channels=256, pass_layers=3,
             adaptive_type="gate")             # 注意：是 'gate'，不是源码默认注释里的 'learned'
# 还要把最后一层 MLP 换成 checkpoint 里的宽度（384→1024→512→128→64→1），
# 否则 load_state_dict 会因形状不匹配失败；然后：
model.load_state_dict(ckpt["model"], strict=True)  # 已是 EMA 权重，不要再套 EMA
model.eval()
print(ckpt["epoch"], ckpt["val_mae"])          # 44  0.0202435…
```

---

## 5. 关键发现：Q / D / p 是解析式算出来的，不需要模型

对 CSV 的 3 000 抽样做回归（`N` = 每克气体产物摩尔数、`M` = 气体产物平均分子量、`Q` = 爆热、`ρ` = 密度）：

| 拟合式 | 平均相对残差 |
| --- | --- |
| `ln D = −0.5735 + 0.4666·ln(N·M^½·Q^½) + 1.3877·ln(1 + 1.30ρ)` | **0.022 %** |
| `ln P = −0.0918 + 0.9332·ln(N·M^½·Q^½) + 2.5115·ln ρ` | **0.043 %** |
| `ln P = −1.2061 + 2.0002·ln D + 0.6806·ln ρ` | **0.068 %** |

残差是数值噪声级 ⇒ **`Q(爆热)`、`D(爆速)`、`p(爆压)` 全部是 `(Formula, ρ, ΔHf)` 的闭式函数**
（Kamlet–Jacobs 一族公式；我用标准气体产物规则的绝对量偏低约 25%，但函数形式确定，
`P ∝ D²·ρ^0.68` 也与 K-J 的 `P = 1.558·ρ²·φ^½` 在 ρ∈[1.2, 2] 内一致）。

**推论（与目录内容完全吻合）**：只要用模型预测 **密度** 与 **生成焓** 两个量，
其余三个爆轰指标直接用公式算即可 —— 这正是 `model_save` 里**只有 `best_density.pt` 和 `best_eof.pt`** 的原因。
→ 做 `wmview.boom` 时**不需要 5 个模型，只需 2 个 + 一段公式**。

---

## 6. 问题清单

### 6.1 阻塞级（当前状态跑不起来）

| # | 位置 | 问题 |
| --- | --- | --- |
| 1 | `dataset/data_load.py:31` | 硬编码 `/export/home/zhengym/model/fusion/GRMP/data/new_data/pt/123679_49_new.pt`，**该文件未随包提供** |
| 2 | `data/features.py:401-407` | 硬编码 csv、`gdb_xyz` 结构目录、输出 `.pt` 路径；`structure_dir` 未随包提供 |
| 3 | `fea_den.py:80-82`、`fea_eof.py` 同 | `save_dir/log_dir/save_path` 默认值是 HPC 路径（调用时未覆盖 → 会在 Windows 上建出 `E:\export\home\...`） |
| 4 | `fea_*.py`、`features.py`、`data_load.py` | 全部**模块级直接执行**（无 `if __name__ == '__main__'`），`import dataset.data_load` 就会去读不存在的 `.pt` |
| 5 | 全局 | 推理栈极重：rdkit + pymatgen + mendeleev + sklearn + sympy + scipy + torch + torch_geometric + **torch_scatter / torch_sparse / torch_cluster**（C++ 扩展，跨版本 wheel 最容易炸） |

### 6.2 逻辑与正确性

| # | 位置 | 问题 |
| --- | --- | --- |
| 6 | `data/features.py:227,262` | 元素索引 off-by-one：`n = number - 1` 去索引 `ele_nums = [1,3,4,…,9]`，于是 **C 取到 N 的行、N 取 O、O 取 F**。对 CHNO 仍是双射（所以不影响现有精度），但**含 F 分子直接 `IndexError`**，B/Li 也取错行 |
| 7 | `data/features.py:227` + mendeleev | 49 维描述子是**推理时现算**的：列序 = sklearn `OneHotEncoder` 对各物理量唯一值的排序。mendeleev 数据一更新就可能**静默错位**，权重立刻失效 → 必须把这张表固化成 npy/json 随权重发布 |
| 8 | `data/features.py`（原子数校验处） | 只校验 `len(z) == MOL.GetNumAtoms()`，**没有逐原子比对元素是否一致**。xyz 原子序与 RDKit 加 H 后的编号不一致时会静默产生"错配的键 + 坐标" |
| 9 | `layer/geom.py:53` | `rbf_feat = self.lin_rbf(rbf_f)` 之后又 `self.lin_rbf(rbf_feat)` —— 同一线性层连用两次，可疑 |
| 10 | `utils/special.py`（`get_geometry_indices` 末） | `angle_idx_map = match.max(dim=1)[1]` 用 O(Q×M×3) 布尔全匹配；**没有匹配时静默取 0 号角度**；批量大分子会爆内存（本数据集 8–10 重原子无碍） |
| 11 | `model/mix.py:110` | `layer_att` 存了权重却从不调用；`global_max_pool`、`BaseMessagePassing`、`torsion_emb` 等均为未使用代码 |
| 12 | `data/features.py:190` | `atomic_number_map` 只到 Ca 且未知元素**静默丢弃**，错误信息不友好 |

### 6.3 工程门面

- 目录名 `模型/`（中文）不利于打包与跨平台；建议改 `model/`。
- 根目录 32 MB `模型.zip` 与目录内容重复，应删除或 gitignore。
- `fea_den.py` / `fea_eof.py` 应合并为 `--target density|hof`。
- 缺 `requirements.txt` / 环境说明；缺随机种子之外的训练日志（`log/`、`picts/` 都没带过来）。

---

## 7. 复用/移植路线（做成 wmview.boom）

宿主侧现成能力：`wmapi_scene.get_mole_geom(name) → [[元素, x, y, z], …]`（场景当前分子坐标）、
`wmapi_filelist.get_file_list() / load_file()`（读文件）。
注意 **宿主没有 RDKit**（`wmview_app/src-tauri/Cargo.toml` 只有 `ndarray`，无任何化学库），
所以键级/芳香性/隐式氢要么插件自带，要么用距离判据自建键图。

**路线 A：把原模型搬进宿主 Python 运行时**（类似 `wmview.xyzrender` 内置 Python）

- 优点：精度与原实现完全等价，改动最小。
- 代价：要携带 11 个依赖（torch CPU 数百 MB + 三个 C++ 扩展），**仅 Windows**；
  仍需先做 §6.1 的 1–5 项改造（去硬编码、去 import 副作用、固化元素表）。

**路线 B（推荐）：把推理重写成 Rust/WASM**

- 网络里没有不可移植算子：MLP、sin、Bessel 零点、球谐、`scatter_add`、LayerNorm 而已。
- `utils/special.py` 的 sympy 现场推导 → 换成**预计算的数值系数表**（Bessel 零点、球谐系数）。
- 49 维元素描述子 → 常量表；键图 → 距离判据。
- ≤17 个原子、3.8 M 参数的前向在 WASM 里几乎无开销，且能像 `wmview.flavor` 一样**支持安卓**。
- 代价：需要数值对齐测试 —— 建议用原 PyTorch 在同一批 xyz 上生成 golden 输出做回归。

**发布侧待办**：`package.json` / `info.json` / `pack.py` / `src/wmapi.d.ts` 全缺；
`plugins/update.py` 依赖 `info.json` 自动发现插件；两个权重合计 30 MB，
需决定是打进插件包还是发布到 OSS 由插件首次运行拉取。

---

## 8. 环境（原项目侧）

```
python 3.12（.pyc 里是 cpython-312）
torch / torch_geometric / torch_scatter / torch_sparse / torch_cluster
rdkit, pymatgen, mendeleev, scikit-learn, sympy, scipy, pandas, numpy
tensorboard (SummaryWriter), tqdm, matplotlib
训练设备：cuda:0（checkpoint storage 标记为 cuda:0）
```

---

## 9. 待确认

1. `wmview.boom` 的目标是否就是"含能材料 ρ / ΔHf / 爆轰性能预测"？→ 直接决定"只需 2 个模型 + 公式"（§5）。
2. 输入形态是 **SMILES**、**xyz**，还是直接用场景里已加载的分子？→ 决定是否要自带键级感知。
3. 走 **Python 捆包**（Windows-only，数百 MB）还是 **Rust/WASM 重写**（可上安卓，需数值对齐）？
4. 是否执行这些低风险整理：删 `模型.zip`、`模型/` → `model/`、导出固化元素描述子表、
   修复 `n = number - 1` 元素索引、合并两份训练脚本、把「ρ + ΔHf → Q/D/p」的公式落成代码。
