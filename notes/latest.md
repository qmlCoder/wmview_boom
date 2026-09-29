- 初始化: 在 plugins/wmview.boom 下建立插件工程（Vue3 + Module Federation remote，name=wmview_boom）
- 新增: runtime/ 纯 numpy 推理运行时（grmp_numpy/geometry/basis_gen/features/perf + boom_infer.py 入口）
- 新增: tools/ 开发工具（gen_basis / freeze_element_table / export_weights / fit_perf / make_icon / prepare_python / verify_torch / check_golden / _torch_stubs）
- 新增: 前端面板 Plugin.vue（选分子/读取范围/键图来源/元素索引/模型开关 + 结果表 + 日志），对外 API window.wmapi_boom
- 发现: checkpoint 与 模型/model/mix.py 源码不是同一版 —— 权重里 MLP 是 384→1024→512→128→64→1，融合层是 adaptive_type='gate'（gate_net.*），源码写的是 384→512→256… 与 'learned'；按权重真实结构重建后 load_state_dict(strict=True) 通过
- 修复: 残差直连 res_x 语义（应为整个层的输入 x，不是 mlp_x2 的输出），修正前 density 偏差约 5%、eof 偏差可达 55%
- 保留: 原实现元素索引 off-by-one（n = 原子序数 - 1）必须原样复现，默认 index_mode=legacy；另提供 corrected 仅作对照
- 固化: 49 维元素描述子表（mendeleev + OneHotEncoder 现算 → element_table.json），避免上游数据更新静默错位
- 固化: sympy 现场推导的 Bessel/球谐基组 → runtime/basis_gen.py（含 float32 零点复现）
- 验证: tools/verify_torch.py 用原始 PyTorch 代码（打 torch_cluster/scatter/sparse/geometric 最小替身）与 numpy 实现逐分子比对，8 分子 × 2 模型最大相对偏差 1.6e-6
- 验证: tools/check_golden.py（不依赖 torch，内置 python 可跑）回归通过
- 验证: 经主程序 HTTP run_script → wmapi_cores.run_exe 跑内置 python + read_text 读回 result.json，端到端通过（token/ρ/ΔHf/Q/D/p 正确）
- 新增: Q/D/p 经验估算（tools/fit_perf.py 在随包 CSV 上回归，留出集 Q 1.1% / D 4.2% / p 5.4%），UI 明确标注"经验估算，非模型输出"
- 新增: pack.py 打包（dist + runtime + python-3.10 + info.json + icon.png），产出 wmview.boom.zip（约 50MB）
- 精简: prepare_python.py 拷贝内置 python 后删除 pip/setuptools/wheel/get-pip/Scripts，包体积 55.7MB → 50.7MB
- 待办: 上架前补 github 地址、传 icon.png 到 OSS、在 plugins/update.py 里登记（update.py 依赖 info.json 自动发现插件）
- 修复: 覆盖域警告文案与判据 —— 原来把「本分子不含 N/O」写成"N 原子数 0 超出训练集范围 [1,5]"，
  容易被当成误报；改为按数据集实测事实说明：训练集 100% 分子都含 C/N/O（无 N-free / O-free 样本）、
  重原子数只有 8~10、密度实测范围 [1.2167,1.957]；H 的范围修正为 [0,8]（有 8 条 H=0 的样本），
  并新增重原子数检查（C2H4/N2/CO2 这类小分子会明确提示"超出模型见过的分子规模"）

- 移植: 随包推理运行时由「numpy + 内置 python 3.10（外部进程）」整体换成 **Rust→wasm 推理核**
  新增 boom-core/（2644 行 Rust，编出 284 KB wasm）：weights/table/graph/geometry/basis/model/perf/pipeline
  + wasm.rs（wasm-bindgen 导出）+ main.rs（原生 CLI：--golden 对拍 / --check / --bench）
  新增 boom-data/：扁平权重（density/eof 各 226 张量 / 3805320 参数 / 15.22 MB f32，另存 fp16 对照）
  新增 tools/export_flat_weights.py（npz → 索引 json + f32 bin）
- 新增: Q/D/p 第二套实现 kamlet_noco（无 CO 氧平衡 + 对数-对数 Kamlet）：留出集 D 4.194%→2.313%、
  p 5.348%→4.624%；两套都实现，UI/API 可切换，另一套放在结果 derived_alt 里对照
- 修复: tools/fit_perf.py 改为不依赖 pandas（stdlib csv + numpy），并同时产出
  runtime/weights/perf_coeff.json + boom-data/perf_coeff.json + perf_coeff_kamlet.json；
  重拟合的 linear14 系数与移植前**逐项 bit 相同**
- 新增: tools/check_basis_parity.py 门禁（basis.rs 与 gen_basis.py 产物的长常量必须完全一致，62/62）
- 新增: tools/compare_impls.py（numpy 参照 vs Rust 逐字段交叉核对，改用 sys.executable）
- 性能: f64 标量 368 ms → f32 标量 264 ms → 手写 wasm SIMD 点积 **82 ms**（17 原子分子，两个模型 + Q/D/p）；
  原生 29 ms。LLVM 对归约循环不会自动向量化（开 +simd128 只快 15%），必须手写 v128
- 修复: SIMD 只能由 boom-core/build-wasm.mjs 显式设 RUSTFLAGS 打开 —— wasm-pack 自己会设 RUSTFLAGS，
  与 .cargo/config.toml 的 [target.*] rustflags 互斥，写 config 里会被静默忽略（实测二进制字节相同）
- 修复: vite.config.ts 补 `base: './'`（wasm 与权重是 ?url 外部资源，绝对 /assets/ 在宿主里 404）；
  原注释里「外链 assets 取不到」的判断是错的，真正原因是缺相对 base
- 变更: 前端不再用 run_exe/read_text/结果文件/token；src/boom/runtime.ts 用 ?url + initSync + load_model；
  面板新增「Q/D/p 口径」与推理核状态；device 加 android、info.json platform = windows,android
- 删除: python-3.10/（65 MB）与 tools/prepare_python.py；随包不再有任何 Python
  （tools/ 下的生成脚本保持 Python，它们不进包）
- 保留: runtime/*.py 作为迁移期的 numpy 参照实现（不进包），用于和 Rust 交叉核对
- 验证: Rust 原生对拍原 PyTorch 参考输出 最大相对偏差 **1.818e-6**（阈值 1e-4）
- 验证: wasm 链路对拍 **2.119e-6**；numpy 参照对拍 2.14e-6；Rust vs numpy 参照数值 ≤7.9e-7 且
  formula/原子数/键数/bond_mode/警告文本逐字一致
- 验证: 宿主端到端 —— 装进 plugs/ 启动主程序，经 HTTP run_script 调 wmapi_boom.predict_result，
  8/8 分子与原生 CLI 一致（ρ 差 ≤3.2e-8、ΔHf 差 0、p 差 ≤9.7e-7）；U/Cr 等不支持元素干净返回 ok:false
- 发布: 包体积 **48.4 MB → 27.9 MB**（−42%），版本 0.1.0 → 0.2.0
- 待办: 上架前补 github 地址、传 icon.png 到 OSS（update.py 依赖 info.json 自动发现插件）
