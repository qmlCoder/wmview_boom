//! 几何基组，逐行对应 `runtime/basis_gen.py`（= `tools/gen_basis.py` 的产物）。
//!
//! `runtime/basis_gen.py` 把 `模型/utils/special.py` 里 sympy 现场推导的
//! Bessel 零点/归一化系数与实球谐固化成数值代码；本文件是同一批数值的 Rust 版本。
//!
//! **全部用 f64**（原实现也是 f64 算完再 `.astype(f32)`）：基组里的 sin/cos 与
//! `1/x` 对 float32 很敏感，降精度会直接把对拍误差顶到 1e-4 以上。

pub const NUM_SPHERICAL: usize = 3;
pub const NUM_RADIAL: usize = 6;
pub const CUTOFF: f64 = 5.0;
pub const ENVELOPE_EXPONENT: i32 = 5;

/// 多项式包络：`1/x + a x^p + b x^(p+1) + c x^(p+2)`（x>=1 时为 0）。
/// 对应 `runtime/grmp_numpy.py:envelope`。
pub fn envelope(x: f64) -> f64 {
    if x < 1.0 {
        let p = ENVELOPE_EXPONENT as f64;
        let a = -(p + 1.0) * (p + 2.0) / 2.0;
        let b = p * (p + 2.0);
        let c = -p * (p + 1.0) / 2.0;
        1.0 / x + a * x.powi(ENVELOPE_EXPONENT) + b * x.powi(ENVELOPE_EXPONENT + 1)
            + c * x.powi(ENVELOPE_EXPONENT + 2)
    } else {
        0.0
    }
}

/// 归一化球贝塞尔基组，单个距离 -> 18 维（`[NUM_SPHERICAL][NUM_RADIAL]` 行主序）。
pub fn sbf_radial_one(x: f64) -> [f64; 18] {
    let mut o = [0f64; 18];
    o[0] = 1.4142135784861405 * (3.1415927410125732 * x).sin() / x;
    o[1] = 1.4142135784861405 * (6.2831854820251465 * x).sin() / x;
    o[2] = 1.4142135037687432 * (9.42477798461914 * x).sin() / x;
    o[3] = 1.4142135784861406 * (12.566370964050293 * x).sin() / x;
    o[4] = 1.4142135388886006 * (15.707962989807129 * x).sin() / x;
    o[5] = 1.4142135622649037 * (18.84955596923828 * x).sin() / x;
    o[6] = (-1.448811866318273 * x * (4.493409633636475 * x).cos()
        + 0.3224304001738124 * (4.493409633636475 * x).sin())
        / (x * x);
    o[7] = (-1.426012656427086 * x * (7.7252516746521 * x).cos()
        + 0.18459109379000333 * (7.7252516746521 * x).sin())
        / (x * x);
    o[8] = (-1.4201481479569723 * x * (10.904121398925781 * x).cos()
        + 0.13023957602827846 * (10.904121398925781 * x).sin())
        / (x * x);
    o[9] = (-1.4177827892570553 * x * (14.066193580627441 * x).cos()
        + 0.10079363554399578 * (14.066193580627441 * x).sin())
        / (x * x);
    o[10] = (-1.4165958636772314 * x * (17.220754623413086 * x).cos()
        + 0.082260963276908224 * (17.220754623413086 * x).sin())
        / (x * x);
    o[11] = (-1.4159164903648997 * x * (20.37130355834961 * x).cos()
        + 0.069505443591731095 * (20.37130355834961 * x).sin())
        / (x * x);
    o[12] = (-1.482208216540892 * (x * x) * (5.763459205627441 * x).sin()
        - 0.7715201046761975 * x * (5.763459205627441 * x).cos()
        + 0.1338640696758789 * (5.763459205627441 * x).sin())
        / (x * x * x);
    o[13] = (-1.440543582577989 * (x * x) * (9.095011711120605 * x).sin()
        - 0.47516494590653965 * x * (9.095011711120605 * x).cos()
        + 0.052244566692041579 * (9.095011711120605 * x).sin())
        / (x * x * x);
    o[14] = (-1.4283878612362195 * (x * x) * (12.322940826416016 * x).sin()
        - 0.34773871302885647 * x * (12.322940826416016 * x).cos()
        + 0.028218808961853324 * (12.322940826416016 * x).sin())
        / (x * x * x);
    o[15] = (-1.4231083282212727 * (x * x) * (15.514602661132812 * x).sin()
        - 0.27518107153071555 * x * (15.514602661132812 * x).cos()
        + 0.017736907450430507 * (15.514602661132812 * x).sin())
        / (x * x * x);
    o[16] = (-1.4203255929218544 * (x * x) * (18.689035415649414 * x).sin()
        - 0.22799340276264874 * x * (18.689035415649414 * x).cos()
        + 0.012199313538233045 * (18.689035415649414 * x).sin())
        / (x * x * x);
    o[17] = (-1.418676078162404 * (x * x) * (21.85387420654297 * x).sin()
        - 0.194749370032201 * x * (21.85387420654297 * x).cos()
        + 0.008911434567235396 * (21.85387420654297 * x).sin())
        / (x * x * x);
    o
}

/// 实球谐（m=0，l=0..2），单个键角 -> 3 维。
pub fn sph_sbf_one(theta: f64) -> [f64; 3] {
    let c = theta.cos();
    [0.282094791773878, 0.48860251190292 * c, 0.94617469575756 * (c * c) - 0.31539156525252]
}

/// 实球谐（l=0..2，全部 m），单个 (键角, 二面角) -> 9 维。
pub fn sph_tbf_one(theta: f64, phi: f64) -> [f64; 9] {
    let ct = theta.cos();
    let st = theta.sin();
    let cp = phi.cos();
    let sp = phi.sin();
    [
        0.282094791773878,
        0.48860251190292 * ct,
        -0.48860251190292 * st * cp,
        -0.48860251190292 * sp * st,
        0.94617469575756 * (ct * ct) - 0.31539156525252,
        0.27313710764802 * (phi - 2.0 * theta).sin() - 0.27313710764802 * (phi + 2.0 * theta).sin(),
        0.54627421529604 * (st * st) * (2.0 * phi).cos(),
        1.09254843059208 * sp * (st * st) * cp,
        -0.27313710764802 * (phi - 2.0 * theta).cos() + 0.27313710764802 * (phi + 2.0 * theta).cos(),
    ]
}

/// `BesselBasisLayer`：`envelope(d/c) * sin(freq * d/c)`，`(E,6)`。
/// `freq` 是可学习参数（`rbf.freq`），从权重里读、f32 存储后转 f64。
pub fn rbf_basis(dists: &[f64], freq: &[f32]) -> Vec<f64> {
    let mut out = vec![0f64; dists.len() * NUM_RADIAL];
    for (e, d) in dists.iter().enumerate() {
        let rs = *d / CUTOFF;
        let env = envelope(rs);
        for r in 0..NUM_RADIAL {
            out[e * NUM_RADIAL + r] = env * (freq[r] as f64 * rs).sin();
        }
    }
    out
}

/// `SphericalBasisLayer`：`(T, 18)`，`T = idx_kj.len()`。
pub fn sbf_basis(dists: &[f64], angles: &[f64], idx_kj: &[usize]) -> Vec<f64> {
    let e = dists.len();
    // rbf = sbf_radial(rs) * envelope(rs)[:, None]
    let mut rbf = vec![0f64; e * NUM_SPHERICAL * NUM_RADIAL];
    for k in 0..e {
        let rs = dists[k] / CUTOFF;
        let env = envelope(rs);
        let row = sbf_radial_one(rs);
        for i in 0..(NUM_SPHERICAL * NUM_RADIAL) {
            rbf[k * NUM_SPHERICAL * NUM_RADIAL + i] = row[i] * env;
        }
    }
    let mut out = vec![0f64; idx_kj.len() * NUM_SPHERICAL * NUM_RADIAL];
    for (t, &eidx) in idx_kj.iter().enumerate() {
        let cbf = sph_sbf_one(angles[t]);
        for sp in 0..NUM_SPHERICAL {
            for r in 0..NUM_RADIAL {
                out[t * NUM_SPHERICAL * NUM_RADIAL + sp * NUM_RADIAL + r] =
                    rbf[eidx * NUM_SPHERICAL * NUM_RADIAL + sp * NUM_RADIAL + r] * cbf[sp];
            }
        }
    }
    out
}

/// `torsionLayer`：`(Q, 54)`，**注意没有 envelope**（与原实现一致）。
pub fn tbf_basis(
    dists: &[f64],
    angles: &[f64],
    torsions: &[f64],
    idx_kj_t: &[usize],
    angle_idx_map: &[usize],
) -> Vec<f64> {
    let e = dists.len();
    let mut rbf = vec![0f64; e * NUM_SPHERICAL * NUM_RADIAL];
    for k in 0..e {
        let row = sbf_radial_one(dists[k] / CUTOFF);
        rbf[k * NUM_SPHERICAL * NUM_RADIAL..(k + 1) * NUM_SPHERICAL * NUM_RADIAL]
            .copy_from_slice(&row);
    }
    let dim = NUM_SPHERICAL * NUM_SPHERICAL * NUM_RADIAL; // 54
    let mut out = vec![0f64; idx_kj_t.len() * dim];
    for q in 0..idx_kj_t.len() {
        let eidx = idx_kj_t[q];
        let cbf = sph_tbf_one(angles[angle_idx_map[q]], torsions[q]);
        for a in 0..NUM_SPHERICAL {
            for b in 0..NUM_SPHERICAL {
                for r in 0..NUM_RADIAL {
                    out[q * dim + a * NUM_SPHERICAL * NUM_RADIAL + b * NUM_RADIAL + r] =
                        rbf[eidx * NUM_SPHERICAL * NUM_RADIAL + b * NUM_RADIAL + r]
                            * cbf[a * NUM_SPHERICAL + b];
                }
            }
        }
    }
    out
}
