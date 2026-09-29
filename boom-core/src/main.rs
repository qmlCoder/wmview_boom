//! 原生 CLI：不依赖 wasm，用来做数值对拍与自检。
//!
//! ```text
//! boom-core --data ../boom-data --golden ../tools/golden.json   # 与 PyTorch 参考对拍
//! boom-core --data ../boom-data --check                         # 环境/结构自检
//! boom-core --data ../boom-data < payload.json                  # 单分子（等价 boom_infer.py）
//! ```
//!
//! `--golden` 是移植的**验收门禁**：`tools/golden.json` 里的 `density` / `eof` 是原始
//! PyTorch 模型的输出，最大相对偏差必须 < 1e-4（与 `tools/check_golden.py` 同一阈值）。

use std::io::Read;
use std::path::{Path, PathBuf};

use boom_core::perf::{KamletPerf, LinearPerf, Variant};
use boom_core::pipeline::{Predictor, Request};
use boom_core::table::ElementTable;
use serde::Deserialize;

const TOL: f64 = 1e-4;

#[derive(Deserialize)]
struct GoldenMol {
    name: String,
    syms: Vec<String>,
    xyzs: Vec<[f64; 3]>,
    bonds: Vec<[i64; 2]>,
    density: f32,
    eof: f32,
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("读不到 {}：{e}", path.display()))
}

fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|e| format!("读不到 {}：{e}", path.display()))
}

fn load(data_dir: &Path) -> Result<Predictor, String> {
    let table = ElementTable::from_json(&read_text(&data_dir.join("element_table.json"))?)?;
    let mut models = Vec::new();
    for target in ["density", "eof"] {
        let index = read_text(&data_dir.join(format!("{target}.json")))?;
        let bin = read(&data_dir.join(format!("{target}.bin")))?;
        let m = boom_core::load_model(&index, &bin)?;
        // 诊断信息走 stderr：stdout 留给「单分子模式的裸 JSON」
        eprintln!(
            "{target}: {} 个张量 / {} 参数（{:.2} MB）epoch={:?} val_mae={:?}",
            m.weights().tensor_count(),
            m.weights().numel(),
            m.weights().numel() as f64 * 4.0 / 1024.0 / 1024.0,
            m.weights().meta().epoch,
            m.weights().meta().val_mae,
        );
        models.push(m);
    }
    let eof = models.pop().expect("eof");
    let density = models.pop().expect("density");
    eprintln!(
        "结构：隐层 {} / 输入 {} / pass_layers {}",
        density.hidden, density.in_dim, density.n_layers
    );

    let linear: LinearPerf =
        serde_json::from_str(&read_text(&data_dir.join("perf_coeff.json"))?)
            .map_err(|e| format!("Q/D/p 线性系数解析失败：{e}"))?;
    let kamlet: KamletPerf =
        serde_json::from_str(&read_text(&data_dir.join("perf_coeff_kamlet.json"))?)
            .map_err(|e| format!("Kamlet 系数解析失败：{e}"))?;

    Ok(Predictor {
        table,
        density,
        eof,
        linear,
        kamlet,
    })
}

/// 硝基甲烷 CH3NO2 的近似几何（与 `runtime/boom_infer.py:CHECK_MOL` 一致）。
fn check_mol() -> Request {
    serde_json::from_str(
        r#"{
          "name": "selftest(CH3NO2)",
          "syms": ["C","N","O","O","H","H","H"],
          "xyzs": [[0.0,0.0,0.0],[1.47,0.0,0.0],[2.08,1.06,0.0],[1.90,-1.10,0.0],
                   [-0.55,0.87,0.0],[-0.55,-0.48,0.82],[-0.55,-0.48,-0.82]],
          "bond_mode": "distance",
          "targets": ["density","eof"],
          "derived": true
        }"#,
    )
    .expect("自检分子 JSON 有误")
}

fn run_check(pr: &Predictor, variant: Variant) -> Result<(), String> {
    let mut req = check_mol();
    req.perf_variant = Some(variant.as_str().to_string());
    let res = pr.predict(&req)?;
    println!("\n=== 环境自检（硝基甲烷 CH3NO2） ===");
    println!(
        "分子 {}  式 {}  原子 {}（重原子 {}）  键 {}（{}）",
        res.name, res.formula, res.n_atoms, res.n_heavy, res.n_bonds, res.bond_mode
    );
    if let Some(rho) = res.results.density {
        let m = res.meta.as_ref().and_then(|m| m.get("density"));
        println!(
            "  密度 ρ      = {rho:.4} g/cm^3   （epoch {:?}，val MAE {:?}）",
            m.and_then(|x| x.epoch),
            m.and_then(|x| x.val_mae)
        );
    }
    if let Some(eof) = res.results.eof {
        let m = res.meta.as_ref().and_then(|m| m.get("eof"));
        println!(
            "  生成焓 ΔHf  = {eof:.3} kcal/mol   （epoch {:?}，val MAE {:?}）",
            m.and_then(|x| x.epoch),
            m.and_then(|x| x.val_mae)
        );
    }
    if let Some(d) = res.derived {
        println!(
            "  经验估算（{}）：Q = {:.1} cal/g，D = {:.3} km/s，p = {:.2} GPa",
            res.perf_variant.as_deref().unwrap_or("?"),
            d.q,
            d.d,
            d.p
        );
    }
    if let Some(a) = res.derived_alt {
        println!(
            "  另一套对照：Q = {:.1} cal/g，D = {:.3} km/s，p = {:.2} GPa",
            a.q, a.d, a.p
        );
    }
    for w in &res.warnings {
        println!("  [警告] {w}");
    }
    Ok(())
}

fn run_golden(pr: &Predictor, path: &Path) -> Result<(), String> {
    let text = read_text(path)?;
    let mols: Vec<GoldenMol> =
        serde_json::from_str(&text).map_err(|e| format!("golden 解析失败：{e}"))?;
    println!("\n=== 与原始 PyTorch 参考输出对拍（{} 个分子 × 2 个模型） ===", mols.len());

    let mut worst = 0f64;
    let mut worst_at = String::new();
    for m in &mols {
        let req = Request {
            name: m.name.clone(),
            syms: m.syms.clone(),
            xyzs: m.xyzs.clone(),
            bonds: Some(m.bonds.clone()),
            bond_mode: Some("host".to_string()),
            index_mode: Some("legacy".to_string()),
            targets: None,
            derived: Some(false),
            perf_variant: None,
            token: None,
        };
        let res = pr.predict(&req)?;
        let got_d = res.results.density.ok_or("缺少 density 结果")?;
        let got_e = res.results.eof.ok_or("缺少 eof 结果")?;
        let rd = rel(got_d, m.density);
        let re = rel(got_e, m.eof);
        for (tag, r) in [("density", rd), ("eof", re)] {
            if r > worst {
                worst = r;
                worst_at = format!("{} {}", m.name, tag);
            }
        }
        println!(
            "  {:>16}  N={:2}  密度 torch={:12.6} rust={:12.6} 相对差={:.2e}   \
             ΔHf torch={:12.6} rust={:12.6} 相对差={:.2e}",
            m.name,
            m.syms.len(),
            m.density,
            got_d,
            rd,
            m.eof,
            got_e,
            re
        );
    }
    println!("\n最大相对偏差 {worst:.3e}（阈值 {TOL:.0e}，最大处：{worst_at}）");
    if worst < TOL {
        println!("判定：通过（Rust 实现与原 PyTorch 模型在 float32 精度内一致）");
        Ok(())
    } else {
        Err(format!("判定：不通过，最大相对偏差 {worst:.3e} ≥ {TOL:.0e}"))
    }
}

fn rel(got: f32, want: f32) -> f64 {
    (got as f64 - want as f64).abs() / (want as f64).abs().max(1e-6)
}

/// 单分子耗时基准（权重只加载一次，纯前向）。
fn run_bench(pr: &Predictor, n: usize, variant: Variant) -> Result<(), String> {
    println!("\n=== 前向耗时（硝基甲烷 CH3NO2，{n} 次取平均） ===");
    for (tag, targets) in [
        ("两个模型 + Q/D/p", vec!["density".to_string(), "eof".to_string()]),
        ("仅密度模型", vec!["density".to_string()]),
        ("仅生成焓模型", vec!["eof".to_string()]),
    ] {
        let mut req = check_mol();
        req.targets = Some(targets);
        req.perf_variant = Some(variant.as_str().to_string());
        pr.predict(&req)?; // 预热
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            pr.predict(&req)?;
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  {tag:20} {per:8.3} ms/分子");
    }
    Ok(())
}

fn run_single(pr: &Predictor, variant: Variant) -> Result<(), String> {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .map_err(|e| e.to_string())?;
    let mut req: Request =
        serde_json::from_str(&raw).map_err(|e| format!("输入 JSON 解析失败：{e}"))?;
    if req.perf_variant.is_none() {
        req.perf_variant = Some(variant.as_str().to_string());
    }
    let res = pr.predict(&req).unwrap_or_else(|e| {
        boom_core::pipeline::BoomResult::failure(&req.name, e)
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&res).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("boom-data");
    let mut golden: Option<PathBuf> = None;
    let mut check = false;
    let mut bench: Option<usize> = None;
    let mut variant = Variant::Linear14;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                i += 1;
                data_dir = PathBuf::from(args.get(i).expect("--data 需要目录参数"));
            }
            "--golden" => {
                i += 1;
                golden = Some(PathBuf::from(args.get(i).expect("--golden 需要文件参数")));
            }
            "--check" => check = true,
            "--bench" => {
                i += 1;
                bench = Some(
                    args.get(i)
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(200),
                );
            }
            "--variant" => {
                i += 1;
                variant = match Variant::parse(args.get(i).map(|s| s.as_str()).unwrap_or("")) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{e}");
                        std::process::exit(2);
                    }
                };
            }
            other => {
                eprintln!("未知参数：{other}");
                eprintln!(
                    "用法：boom-core [--data <目录>] [--golden <golden.json>] [--check] [--bench N] \
                     [--variant linear14|kamlet]"
                );
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let pr = match load(&data_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let result = if let Some(g) = &golden {
        run_golden(&pr, g)
    } else if let Some(n) = bench {
        run_bench(&pr, n, variant)
    } else if check {
        run_check(&pr, variant)
    } else {
        run_single(&pr, variant)
    };

    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
