//! Q / D / p 的经验估算，对应 `runtime/perf.py`，并额外实现了第二套（Kamlet 形式）。
//!
//! 模型只输出密度 ρ 与固体生成焓 ΔHf；Q(爆热)、D(爆速)、p(爆压) **不是网络输出**
//! （见 模型/README.md §5），这里用 `tools/fit_perf.py` 在随包数据集上拟合的经验关系
//! 做量级估算。两套实现并存，便于对照：
//!
//! - `Linear14`（现行）：14 个特征（元素计数 + ρ + ΔHf + 氧平衡产物）的线性回归；
//! - `KamletNoCO`（候选）：Q 仍用上面的回归，D/p 换成「无 CO 氧平衡 +
//!   对数-对数 Kamlet 形式」——`tools/fit_perf.py` 的留出集实测把 D 的中位误差
//!   从 4.19% 降到 2.31%、p 从 5.35% 降到 4.62%。

use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LinearTarget {
    pub mode: String,
    pub coef: Vec<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LinearPerf {
    pub features: Vec<String>,
    pub targets: HashMap<String, LinearTarget>,
}

/// Kamlet 形式的系数（Q 由 `LinearPerf` 给出）。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct KamletPerf {
    #[serde(default)]
    pub variant: String,
    #[serde(rename = "D")]
    pub d: KamletTarget,
    #[serde(rename = "p")]
    pub p: KamletTarget,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KamletTarget {
    #[serde(default)]
    pub basis: String,
    pub coef: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Linear14,
    KamletNoCO,
}

impl Variant {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "linear14" | "linear" | "" => Ok(Variant::Linear14),
            "kamlet" | "kamlet_noco" => Ok(Variant::KamletNoCO),
            other => Err(format!("未知的 perf_variant：{other}（应为 linear14 / kamlet）")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Linear14 => "linear14",
            Variant::KamletNoCO => "kamlet_noco",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Perf {
    pub q: f32,
    pub d: f32,
    pub p: f32,
}

/// 含 CO 的氧平衡产物规则（`runtime/perf.py:products`）。
pub fn products_with_co(c: f64, h: f64, n: f64, o: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
    let nh2o = h / 2.0;
    let rem = o - nh2o;
    let nco2 = if rem > 0.0 {
        c.min(rem.max(0.0) / 2.0)
    } else {
        0.0
    };
    let rem2 = if rem > 0.0 {
        (rem.max(0.0) - 2.0 * nco2).max(0.0)
    } else {
        0.0
    };
    let nco = if rem > 0.0 { (c - nco2).max(0.0) } else { 0.0 };
    let no2 = rem2 / 2.0;
    let nn2 = n / 2.0;
    let ngas = nh2o + nco2 + nco + no2 + nn2;
    let mass = nh2o * 18.015 + nco2 * 44.01 + nco * 28.01 + no2 * 32.0 + nn2 * 28.014;
    let mgas = if ngas > 0.0 { mass / ngas } else { 0.0 };
    (nco2, nh2o, nco, no2, nn2, ngas, mgas)
}

/// 无 CO 的氧平衡产物规则（Kamlet 形式用的那一套，系数与 模型/README.md §5 相符）。
pub fn products_without_co(c: f64, h: f64, n: f64, o: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
    let nh2o = h / 2.0;
    let rem = o - nh2o;
    let nco2 = if rem > 0.0 {
        c.min(rem.max(0.0) / 2.0)
    } else {
        0.0
    };
    let nco = 0.0;
    let no2 = if rem > 2.0 * c { (rem - 2.0 * c) / 2.0 } else { 0.0 };
    let nn2 = n / 2.0;
    let ngas = nh2o + nco2 + nco + no2 + nn2;
    let mass = nh2o * 18.015 + nco2 * 44.01 + nco * 28.01 + no2 * 32.0 + nn2 * 28.014;
    let mgas = if ngas > 0.0 { mass / ngas } else { 0.0 };
    (nco2, nh2o, nco, no2, nn2, ngas, mgas)
}

/// 现行 14 特征线性回归（`targets` 里没有的量返回 None）。
pub fn linear14(counts: &[(String, usize)], rho: f64, hof: f64, p: &LinearPerf) -> HashMap<String, f64> {
    let get = |k: &str| -> f64 {
        counts
            .iter()
            .find(|(s, _)| s == k)
            .map(|(_, n)| *n as f64)
            .unwrap_or(0.0)
    };
    let (c, h, n, o) = (get("C"), get("H"), get("N"), get("O"));
    let (nco2, nh2o, nco, no2, nn2, ngas, mgas) = products_with_co(c, h, n, o);
    let value = |name: &str| -> f64 {
        match name {
            "bias" => 1.0,
            "C" => c,
            "H" => h,
            "N" => n,
            "O" => o,
            "rho" => rho,
            "hof" => hof,
            "nCO2" => nco2,
            "nH2O" => nh2o,
            "nCO" => nco,
            "nO2" => no2,
            "nN2" => nn2,
            "ngas" => ngas,
            "Mgas" => mgas,
            _ => 0.0,
        }
    };
    let x: Vec<f64> = p.features.iter().map(|f| value(f)).collect();
    let mut out = HashMap::new();
    for (name, t) in &p.targets {
        let mut y: f64 = t.coef.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        if t.mode == "log" {
            y = y.exp();
        }
        out.insert(name.clone(), y);
    }
    out
}

/// 两套实现都算一遍，返回 `(所选, 另一套)`（用于 UI 对照）。
pub fn estimate(
    counts: &[(String, usize)],
    rho: f32,
    hof: f32,
    lin: &LinearPerf,
    kam: &KamletPerf,
    variant: Variant,
) -> (Perf, Perf, Variant) {
    let l = linear14(counts, rho as f64, hof as f64, lin);
    let q = l.get("Q").copied().unwrap_or(0.0);
    let linear = Perf {
        q: q as f32,
        d: l.get("D").copied().unwrap_or(0.0) as f32,
        p: l.get("p").copied().unwrap_or(0.0) as f32,
    };

    // Kamlet：N·M^½·Q^½（N = 每克气体摩尔数、M = 气体平均分子量、Q = 爆热）
    let get = |k: &str| -> f64 {
        counts
            .iter()
            .find(|(s, _)| s == k)
            .map(|(_, n)| *n as f64)
            .unwrap_or(0.0)
    };
    let (c, h, n, o) = (get("C"), get("H"), get("N"), get("O"));
    let (_, _, _, _, _, ngas, mgas) = products_without_co(c, h, n, o);
    let mw = c * 12.011 + h * 1.008 + n * 14.007 + o * 15.999;
    let n_per_g = if mw > 0.0 { ngas / mw } else { 0.0 };
    let k = (n_per_g * mgas.sqrt() * q.max(1.0).sqrt()).max(1e-12).ln();
    let kd = if kam.d.basis.contains("1.30") {
        (1.0 + 1.30 * rho as f64).ln()
    } else {
        (rho as f64).ln()
    };
    let kp = if kam.p.basis.contains("1.30") {
        (1.0 + 1.30 * rho as f64).ln()
    } else {
        (rho as f64).ln()
    };
    let kamlet = Perf {
        q: q as f32,
        d: (kam.d.coef[0] + kam.d.coef[1] * k + kam.d.coef[2] * kd).exp() as f32,
        p: (kam.p.coef[0] + kam.p.coef[1] * k + kam.p.coef[2] * kp).exp() as f32,
    };

    match variant {
        Variant::Linear14 => (linear, kamlet, Variant::KamletNoCO),
        Variant::KamletNoCO => (kamlet, linear, Variant::Linear14),
    }
}
