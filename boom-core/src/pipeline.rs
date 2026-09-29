//! 端到端管线：分子 -> 结果 JSON，对应 `runtime/boom_infer.py` 的 `run()` / `report()` /
//! `domain_warnings()`。产物结构与 `src/plugin-api.d.ts` 的 `BoomResult` 保持一致，
//! 前端只需要换掉「怎么拿到结果」，渲染代码不用动。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::{count_of, formula_counts, hill_formula, prepare};
use crate::model::Model;
use crate::perf::{estimate, KamletPerf, LinearPerf, Perf, Variant};
use crate::table::{ElementTable, IndexMode};

/// 训练集覆盖域（`模型/data/final_123679.csv` 123679 条的实测统计）。
const DOMAIN: &[(&str, f64, f64)] = &[
    ("C", 1.0, 7.0),
    ("H", 0.0, 8.0),
    ("N", 1.0, 5.0),
    ("O", 1.0, 4.0),
    ("heavy", 8.0, 10.0),
    ("atoms", 8.0, 17.0),
    ("rho", 1.2167, 1.9570),
];
const ALWAYS_PRESENT: [&str; 3] = ["C", "N", "O"];
const ELEMENTS: [&str; 4] = ["C", "H", "N", "O"];

fn domain(key: &str) -> (f64, f64) {
    let d = DOMAIN.iter().find(|(k, _, _)| *k == key).expect("缺少覆盖域定义");
    (d.1, d.2)
}

#[derive(Debug, Clone, Deserialize)]
pub struct Request {
    #[serde(default)]
    pub name: String,
    pub syms: Vec<String>,
    pub xyzs: Vec<[f64; 3]>,
    #[serde(default)]
    pub bonds: Option<Vec<[i64; 2]>>,
    #[serde(default)]
    pub bond_mode: Option<String>,
    #[serde(default)]
    pub index_mode: Option<String>,
    #[serde(default)]
    pub targets: Option<Vec<String>>,
    #[serde(default)]
    pub derived: Option<bool>,
    /// `linear14`（现行）或 `kamlet`（候选）；不传按 `linear14`。
    #[serde(default)]
    pub perf_variant: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Results {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eof: Option<f32>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Derived {
    /// 爆热 Q (cal/g)，经验估算
    #[serde(rename = "Q")]
    pub q: f32,
    /// 爆速 D (km/s)，经验估算
    #[serde(rename = "D")]
    pub d: f32,
    /// 爆压 p (GPa)，经验估算
    #[serde(rename = "p")]
    pub p: f32,
}

impl From<Perf> for Derived {
    fn from(v: Perf) -> Self {
        Self {
            q: v.q,
            d: v.d,
            p: v.p,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TargetMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_mae: Option<f64>,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoomResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    pub name: String,
    pub formula: String,
    pub n_atoms: usize,
    pub n_heavy: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counts: Option<BTreeMap<String, usize>>,
    pub bond_mode: String,
    pub n_bonds: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_mode: Option<String>,
    pub results: Results,
    pub derived: Option<Derived>,
    /// 另一套 Q/D/p 实现的结果（对照用，`perf_variant` 之外的那一套）。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_alt: Option<Derived>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perf_variant: Option<String>,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<BTreeMap<String, TargetMeta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl BoomResult {
    pub fn failure(name: &str, error: impl Into<String>) -> Self {
        Self {
            ok: false,
            token: None,
            name: name.to_string(),
            formula: String::new(),
            n_atoms: 0,
            n_heavy: 0,
            counts: None,
            bond_mode: String::new(),
            n_bonds: 0,
            index_mode: None,
            results: Results::default(),
            derived: None,
            derived_alt: None,
            perf_variant: None,
            warnings: Vec::new(),
            meta: None,
            error: Some(error.into()),
        }
    }
}

/// 覆盖域警告（逐条对应 `runtime/boom_infer.py:domain_warnings`）。
pub fn domain_warnings(counts: &[(String, usize)], n_atoms: usize, rho: Option<f32>) -> Vec<String> {
    let mut warns: Vec<String> = Vec::new();

    let mut other: Vec<String> = counts
        .iter()
        .filter(|(s, _)| !ELEMENTS.contains(&s.as_str()))
        .map(|(s, _)| s.clone())
        .collect();
    other.sort();
    other.dedup();
    if !other.is_empty() {
        warns.push(format!(
            "含非 CHNO 元素 {other:?}：远超训练集覆盖域，结果不可信"
        ));
    }

    let absent: Vec<&str> = ALWAYS_PRESENT
        .iter()
        .copied()
        .filter(|el| count_of(counts, el) == 0)
        .collect();
    if !absent.is_empty() {
        warns.push(format!(
            "本分子不含 {}，而训练集里所有分子都含 C/N/O（没有 N-free / O-free 的样本）→ 属外推",
            absent.join("/")
        ));
    }

    let mut over: Vec<String> = Vec::new();
    for el in ELEMENTS {
        let c = count_of(counts, el);
        let (_, hi) = domain(el);
        if c as f64 > hi {
            over.push(format!("{el}={c}（训练集上限 {hi:.0}）"));
        }
    }
    if !over.is_empty() {
        warns.push(format!("原子数超出训练集范围：{}", over.join("，")));
    }

    let n_heavy: usize = ELEMENTS
        .iter()
        .filter(|e| **e != "H")
        .map(|e| count_of(counts, e))
        .sum();
    let (lo, hi) = domain("heavy");
    if (n_heavy as f64) < lo || (n_heavy as f64) > hi {
        warns.push(format!(
            "重原子数 {n_heavy} 超出训练集范围 [{lo:.0},{hi:.0}]（训练集只有 8~10 个重原子的小分子）"
        ));
    }

    let (lo, hi) = domain("atoms");
    if (n_atoms as f64) < lo || (n_atoms as f64) > hi {
        warns.push(format!(
            "原子总数 {n_atoms} 超出训练集范围 [{lo:.0},{hi:.0}]（训练集 8~17 个原子）"
        ));
    }

    if let Some(r) = rho {
        let (lo, hi) = domain("rho");
        if (r as f64) < lo || (r as f64) > hi {
            warns.push(format!(
                "预测密度 {r:.3} 超出训练集实测范围 [{lo:.4},{hi:.4}]，属外推、不要采信"
            ));
        }
    }
    warns
}

/// 推理器：持有固化表、两个模型与两套 Q/D/p 系数，可复用（权重只加载一次）。
pub struct Predictor {
    pub table: ElementTable,
    pub density: Model,
    pub eof: Model,
    pub linear: LinearPerf,
    pub kamlet: KamletPerf,
}

impl Predictor {
    /// 跑一个分子，返回与 Python 版字段一致的结果。
    ///
    /// 注意：默认键图来源 `auto` = 有宿主键就用宿主键，否则按共价半径判据建键
    /// （与 `runtime/boom_infer.py` 一致）。
    pub fn predict(&self, req: &Request) -> Result<BoomResult, String> {
        let index_mode = IndexMode::parse(req.index_mode.as_deref().unwrap_or("legacy"))?;
        let bond_mode = req.bond_mode.clone().unwrap_or_else(|| "auto".to_string());
        let variant = Variant::parse(req.perf_variant.as_deref().unwrap_or("linear14"))?;

        if req.xyzs.len() != req.syms.len() || req.syms.is_empty() {
            return Err("syms / xyzs 形状不匹配".to_string());
        }
        for p in &req.xyzs {
            if p.iter().any(|v| !v.is_finite()) {
                return Err("坐标里有非有限值".to_string());
            }
        }

        let targets: Vec<String> = req
            .targets
            .clone()
            .unwrap_or_else(|| vec!["density".into(), "eof".into()]);
        let want_density = targets.iter().any(|t| t == "density");
        let want_eof = targets.iter().any(|t| t == "eof");
        if !want_density && !want_eof {
            return Err("至少要跑一个模型（density / eof）".to_string());
        }

        let mol = prepare(
            &req.syms,
            &req.xyzs,
            req.bonds.as_deref(),
            &bond_mode,
            index_mode,
            &self.table,
        )?;

        let mut results = Results::default();
        if want_density {
            results.density =
                Some(self.density.forward(&mol.x_geom, &req.xyzs, &mol.j, &mol.i));
        }
        if want_eof {
            results.eof = Some(self.eof.forward(&mol.x_geom, &req.xyzs, &mol.j, &mol.i));
        }

        let counts = formula_counts(&req.syms);
        let n_atoms = req.syms.len();
        // 与原实现一致：n_heavy 统计的是「除 H 以外的全部元素」，
        // 而覆盖域警告里的重原子数只统计 C/N/O（见 domain_warnings）。
        let n_heavy: usize = counts
            .iter()
            .filter(|(s, _)| s != "H")
            .map(|(_, n)| *n)
            .sum();

        let (derived, derived_alt, perf_variant) = match (results.density, results.eof) {
            (Some(rho), Some(hof)) if req.derived.unwrap_or(true) => {
                let (sel, alt, _alt_name) = estimate(
                    &counts,
                    rho,
                    hof,
                    &self.linear,
                    &self.kamlet,
                    variant,
                );
                (
                    Some(Derived::from(sel)),
                    Some(Derived::from(alt)),
                    Some(variant.as_str().to_string()),
                )
            }
            _ => (None, None, None),
        };

        let mut warnings = domain_warnings(&counts, n_atoms, results.density);
        if n_atoms > 0 && count_of(&counts, "H") == 0 {
            warnings.push(
                "分子中没有显式氢原子：原模型是在 AddHs 后的含氢结构上训练的，结果偏差会很大"
                    .to_string(),
            );
        }

        let meta = {
            let mut m = BTreeMap::new();
            for (name, model) in [("density", &self.density), ("eof", &self.eof)] {
                let ix = model.weights().meta();
                m.insert(
                    name.to_string(),
                    TargetMeta {
                        epoch: ix.epoch,
                        val_mae: ix.val_mae,
                        label: ix.label.clone(),
                    },
                );
            }
            Some(m)
        };

        let mut count_map = BTreeMap::new();
        for (k, v) in &counts {
            count_map.insert(k.clone(), *v);
        }

        Ok(BoomResult {
            ok: true,
            token: req.token.clone(),
            name: req.name.clone(),
            formula: hill_formula(&req.syms),
            n_atoms,
            n_heavy,
            counts: Some(count_map),
            bond_mode: mol.bond_mode.to_string(),
            n_bonds: mol.n_bonds,
            index_mode: Some(index_mode.as_str().to_string()),
            results,
            derived,
            derived_alt,
            perf_variant,
            warnings,
            meta,
            error: None,
        })
    }
}
