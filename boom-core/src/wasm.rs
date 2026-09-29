//! wasm 绑定：插件在 WebView 里直接调用，不再需要外部 python 进程。
//!
//! 前端用法：
//! ```js
//! import init, { load_table, load_model, load_perf, predict, is_ready } from 'boom_core.js'
//! initSync({ module: wasmBytes })
//! load_table(elementTableJson)
//! load_model('density', densityIndexJson, new Uint8Array(densityBin))
//! load_model('eof', eofIndexJson, new Uint8Array(eofBin))
//! load_perf(perfCoeffJson, kamletCoeffJson)
//! const json = predict(JSON.stringify({ name, syms, xyzs, bonds, bond_mode, ... }))
//! ```
//!
//! `load_model` 两个模型都到齐后自动组装 `Predictor`（权重只加载一次，之后每个分子
//! 只是几十微秒到毫秒级的前向）。

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::model::Model;
use crate::perf::{KamletPerf, LinearPerf};
use crate::pipeline::{Predictor, Request};
use crate::table::ElementTable;

thread_local! {
    static PARTS: RefCell<Parts> = const { RefCell::new(Parts::new()) };
    static PREDICTOR: RefCell<Option<Predictor>> = const { RefCell::new(None) };
}

struct Parts {
    table: Option<ElementTable>,
    density: Option<Model>,
    eof: Option<Model>,
    linear: Option<LinearPerf>,
    kamlet: Option<KamletPerf>,
}

impl Parts {
    const fn new() -> Self {
        Self {
            table: None,
            density: None,
            eof: None,
            linear: None,
            kamlet: None,
        }
    }
}

/// 四部分都到齐后组装 `Predictor`（只组装一次）。
fn try_build() {
    PARTS.with(|p| {
        let mut p = p.borrow_mut();
        let ready = p.table.is_some()
            && p.density.is_some()
            && p.eof.is_some()
            && p.linear.is_some()
            && p.kamlet.is_some();
        if !ready {
            return;
        }
        let predictor = Predictor {
            table: p.table.take().unwrap(),
            density: p.density.take().unwrap(),
            eof: p.eof.take().unwrap(),
            linear: p.linear.take().unwrap(),
            kamlet: p.kamlet.take().unwrap(),
        };
        PREDICTOR.with(|slot| *slot.borrow_mut() = Some(predictor));
    });
}

/// 加载 49 维元素描述子表（`boom-data/element_table.json` 的文本）。
#[wasm_bindgen]
pub fn load_table(json: &str) -> Result<(), JsValue> {
    let table = ElementTable::from_json(json).map_err(err)?;
    PARTS.with(|p| p.borrow_mut().table = Some(table));
    try_build();
    Ok(())
}

/// 加载一个模型：`target` 为 `density` / `eof`，`bin` 是 f32 小端的扁平权重。
/// 返回参数个数（自检用）。
#[wasm_bindgen]
pub fn load_model(target: &str, index_json: &str, bin: &[u8]) -> Result<usize, JsValue> {
    let model = crate::load_model(index_json, bin).map_err(err)?;
    let n = model.weights().numel();
    PARTS.with(|p| {
        let mut p = p.borrow_mut();
        match target {
            "density" => p.density = Some(model),
            "eof" => p.eof = Some(model),
            other => return Err(err(format!("未知的模型名 {other}（应为 density / eof）"))),
        }
        Ok(())
    })?;
    try_build();
    Ok(n)
}

/// 加载两套 Q/D/p 系数（现行线性回归 + Kamlet 形式）。
#[wasm_bindgen]
pub fn load_perf(linear_json: &str, kamlet_json: &str) -> Result<(), JsValue> {
    let linear: LinearPerf = serde_json::from_str(linear_json).map_err(|e| err(format!("Q/D/p 线性系数解析失败：{e}")))?;
    let kamlet: KamletPerf = serde_json::from_str(kamlet_json).map_err(|e| err(format!("Kamlet 系数解析失败：{e}")))?;
    PARTS.with(|p| {
        let mut p = p.borrow_mut();
        p.linear = Some(linear);
        p.kamlet = Some(kamlet);
    });
    try_build();
    Ok(())
}

/// 全部权重到齐、可以开始预测。
#[wasm_bindgen]
pub fn is_ready() -> bool {
    PREDICTOR.with(|p| p.borrow().is_some())
}

/// 模型自检摘要（JSON）。
#[wasm_bindgen]
pub fn describe() -> String {
    PREDICTOR.with(|p| {
        let b = p.borrow();
        match b.as_ref() {
            Some(pr) => serde_json::json!({
                "ready": true,
                "hidden": pr.density.hidden,
                "in_dim": pr.density.in_dim,
                "pass_layers": pr.density.n_layers,
                "tensors_per_model": pr.density.weights().tensor_count(),
                "params_per_model": pr.density.weights().numel(),
                "density": {
                    "epoch": pr.density.weights().meta().epoch,
                    "val_mae": pr.density.weights().meta().val_mae,
                    "label": pr.density.weights().meta().label,
                },
                "eof": {
                    "epoch": pr.eof.weights().meta().epoch,
                    "val_mae": pr.eof.weights().meta().val_mae,
                    "label": pr.eof.weights().meta().label,
                },
            })
            .to_string(),
            None => serde_json::json!({ "ready": false }).to_string(),
        }
    })
}

/// 单分子预测：入参是 `Request` 的 JSON，返回 `BoomResult` 的 JSON。
///
/// 模型层面的错误（例如坐标不匹配、元素超范围）会以 `{"ok":false,"error":...}`
/// 返回，交给前端按原来的「失败行」渲染；只有「权重还没加载」才抛异常。
#[wasm_bindgen]
pub fn predict(request_json: &str) -> Result<String, JsValue> {
    let req: Request = serde_json::from_str(request_json)
        .map_err(|e| err(format!("请求 JSON 解析失败：{e}")))?;
    PREDICTOR.with(|p| {
        let b = p.borrow();
        let pr = b
            .as_ref()
            .ok_or_else(|| err("推理核尚未就绪：还有权重没加载"))?;
        let out = match pr.predict(&req) {
            Ok(mut r) => {
                r.token = req.token.clone();
                r
            }
            Err(e) => {
                let mut r = crate::pipeline::BoomResult::failure(&req.name, e);
                r.token = req.token.clone();
                r
            }
        };
        serde_json::to_string(&out).map_err(|e| err(format!("结果序列化失败：{e}")))
    })
}

fn err(msg: impl Into<String>) -> JsValue {
    JsValue::from_str(&msg.into())
}
