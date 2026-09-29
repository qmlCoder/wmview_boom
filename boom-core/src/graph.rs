//! 分子 -> 模型输入的「键图」部分，对应 `runtime/features.py`。

use crate::table::{covalent_radius, ElementTable, IndexMode};

/// 元素计数。
pub fn formula_counts(syms: &[String]) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    for s in syms {
        match out.iter_mut().find(|(k, _)| k == s) {
            Some((_, n)) => *n += 1,
            None => out.push((s.clone(), 1)),
        }
    }
    out
}

pub fn count_of(counts: &[(String, usize)], sym: &str) -> usize {
    counts
        .iter()
        .find(|(k, _)| k == sym)
        .map(|(_, n)| *n)
        .unwrap_or(0)
}

/// Hill 式分子式（对应 `runtime/features.py:hill_formula`）。
pub fn hill_formula(syms: &[String]) -> String {
    let mut rest = formula_counts(syms);
    let mut parts: Vec<String> = Vec::new();
    for sym in ["C", "H"] {
        if let Some(pos) = rest.iter().position(|(k, _)| k == sym) {
            let (_, n) = rest.remove(pos);
            parts.push(fmt_part(sym, n));
        }
    }
    rest.sort_by(|a, b| a.0.cmp(&b.0));
    for (s, n) in &rest {
        parts.push(fmt_part(s, *n));
    }
    parts.concat()
}

fn fmt_part(sym: &str, n: usize) -> String {
    if n > 1 {
        format!("{sym}{n}")
    } else {
        sym.to_string()
    }
}

/// 距离判据建键：`d < scale * (r_i + r_j)` 视为成键（无向、去重、i<j）。
/// 对应 `runtime/features.py:distance_bonds`（含它的遍历顺序）。
pub fn distance_bonds(syms: &[String], xyz: &[[f64; 3]], scale: f64) -> Vec<[usize; 2]> {
    let n = xyz.len();
    let mut bonds = Vec::new();
    if n < 2 {
        return bonds;
    }
    for a in 0..n {
        let ra = match covalent_radius(&syms[a]) {
            Some(r) => r,
            None => continue,
        };
        for b in (a + 1)..n {
            let rb = match covalent_radius(&syms[b]) {
                Some(r) => r,
                None => continue,
            };
            let d = dist(xyz[a], xyz[b]);
            if d < scale * (ra + rb) {
                bonds.push([a, b]);
            }
        }
    }
    bonds
}

#[inline]
pub fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// 无向键列表 -> 双向 edge_index `(j, i)`，顺序与 `runtime/features.py` 一致：
/// 前半段来自 `b[:,0]`、后半段来自 `b[:,1]`。
pub fn edge_index_from_bonds(bonds: &[[usize; 2]], n_atoms: usize) -> (Vec<i64>, Vec<i64>) {
    let mut fwd: Vec<i64> = Vec::new();
    let mut bwd: Vec<i64> = Vec::new();
    for b in bonds {
        let (a, c) = (b[0], b[1]);
        if a >= n_atoms || c >= n_atoms || a == c {
            continue;
        }
        fwd.push(a as i64);
        bwd.push(c as i64);
    }
    let mut j = fwd.clone();
    j.extend_from_slice(&bwd);
    let mut i = bwd;
    i.extend_from_slice(&fwd);
    (j, i)
}

/// 建键 + 组装 x_geom 的一步到位版本（原生 CLI 与 wasm 都用它）。
pub struct MolInput {
    pub x_geom: Vec<f32>,
    pub j: Vec<i64>,
    pub i: Vec<i64>,
    pub bond_mode: &'static str,
    pub n_bonds: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn prepare(
    syms: &[String],
    xyz: &[[f64; 3]],
    host_bonds: Option<&[[i64; 2]]>,
    bond_mode: &str,
    index_mode: IndexMode,
    table: &ElementTable,
) -> Result<MolInput, String> {
    let n_atoms = syms.len();
    let (bonds, mode): (Vec<[usize; 2]>, &'static str) = match bond_mode {
        "distance" => (distance_bonds(syms, xyz, 1.25), "distance"),
        "auto" if host_bonds.map(|b| b.is_empty()).unwrap_or(true) => {
            (distance_bonds(syms, xyz, 1.25), "distance")
        }
        "auto" | "host" => {
            let hb = host_bonds.unwrap_or(&[]);
            let v: Vec<[usize; 2]> = hb
                .iter()
                .map(|b| [b[0].max(0) as usize, b[1].max(0) as usize])
                .collect();
            (v, "host")
        }
        other => return Err(format!("未知的 bond_mode：{other}（应为 auto / host / distance）")),
    };
    let (j, i) = edge_index_from_bonds(&bonds, n_atoms);
    let x_geom = table.build_x_geom(syms, index_mode)?;
    Ok(MolInput {
        x_geom,
        j,
        i,
        bond_mode: mode,
        n_bonds: bonds.len(),
    })
}
