//! wmview.boom 推理核（纯 Rust，可编 wasm）。
//!
//! 逐算子对应 `runtime/` 下的 numpy 实现：
//!
//! | 本 crate | numpy 版 |
//! | --- | --- |
//! | `weights` | `tools/export_flat_weights.py` 的扁平权重产物 |
//! | `table` | `runtime/features.py` 的 49 维元素描述子 |
//! | `graph` | `runtime/features.py` 的建键 / edge_index / Hill 式 |
//! | `geometry` | `runtime/geometry.py` |
//! | `basis` | `runtime/basis_gen.py` |
//! | `model` | `runtime/grmp_numpy.py` |
//! | `perf` | `runtime/perf.py`（并多实现一套 Kamlet 形式） |
//! | `pipeline` | `runtime/boom_infer.py` |
//!
//! 数值对齐的验收标准见 `tools/golden.json` + `boom-core --golden`：与原 PyTorch
//! 参考输出的最大相对偏差必须 < 1e-4（float32 前向的噪声上限）。

pub mod basis;
pub mod geometry;
pub mod graph;
pub mod model;
pub mod perf;
pub mod pipeline;
pub mod table;
pub mod weights;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use model::Model;
pub use perf::{KamletPerf, LinearPerf};
pub use pipeline::{BoomResult, Predictor, Request};
pub use table::{ElementTable, IndexMode};
pub use weights::Weights;

/// 读一组权重（索引 json + bin）并做结构自检。
pub fn load_model(index_json: &str, bin: &[u8]) -> Result<Model, String> {
    Model::new(Weights::from_parts(index_json, bin)?)
}
