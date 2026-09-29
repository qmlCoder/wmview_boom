//! 49 维元素描述子表（`boom-data/element_table.json`）。
//!
//! 原实现是推理时用 mendeleev + sklearn 现算 one-hot 的，一旦上游数据更新列序就会
//! 静默错位（见 模型/README.md §6.2 第 7 条），所以固化成常量表随包发布。
//! 本文件的 `row_for` 逐行对应 `runtime/features.py:element_rows`。

/// 原实现用到的 8 个元素（`ele_nums = [1] + list(range(3, 10))`）。
pub const SYMBOLS: [&str; 8] = ["H", "Li", "Be", "B", "C", "N", "O", "F"];

/// 元素符号 -> 原子序数（与 `runtime/features.py:ATOMIC_NUMBER` 一致，含 He）。
fn atomic_number(sym: &str) -> Option<usize> {
    Some(match sym {
        "H" => 1,
        "He" => 2,
        "Li" => 3,
        "Be" => 4,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        _ => return None,
    })
}

/// 共价半径（Å），用于「距离判据」建键。
pub fn covalent_radius(sym: &str) -> Option<f64> {
    Some(match sym {
        "H" => 0.31,
        "He" => 0.28,
        "Li" => 1.28,
        "Be" => 0.96,
        "B" => 0.84,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        _ => return None,
    })
}

/// 元素描述子索引方式。`Legacy` 复现原实现的 off-by-one（必须保留），`Corrected` 只作对照。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    Legacy,
    Corrected,
}

impl IndexMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "legacy" | "" => Ok(IndexMode::Legacy),
            "corrected" => Ok(IndexMode::Corrected),
            other => Err(format!("未知的 index_mode：{other}（应为 legacy / corrected）")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            IndexMode::Legacy => "legacy",
            IndexMode::Corrected => "corrected",
        }
    }
}

#[derive(serde::Deserialize)]
struct RawTable {
    symbols: Vec<String>,
    table: Vec<Vec<f64>>,
}

/// 固化的 49 维描述子表。
pub struct ElementTable {
    /// 8 × 49，行主序，f32（与 `runtime/features.py:build_x_geom` 的 `.astype(DT)` 一致）。
    table: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl ElementTable {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let raw: RawTable =
            serde_json::from_str(text).map_err(|e| format!("元素描述子表解析失败：{e}"))?;
        let rows = raw.table.len();
        let cols = raw.table.first().map(|r| r.len()).unwrap_or(0);
        if rows != 8 || cols != 49 {
            return Err(format!("元素描述子表形状异常：({rows}, {cols})，应为 (8, 49)"));
        }
        if raw.symbols != SYMBOLS {
            return Err(format!(
                "元素描述子表的元素顺序异常：{:?}，应为 {:?}",
                raw.symbols, SYMBOLS
            ));
        }
        let table: Vec<f32> = raw.table.iter().flatten().map(|v| *v as f32).collect();
        Ok(Self { table, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// 元素符号 -> 描述子表行号。
    ///
    /// `legacy`（默认）：`row = Z - 1`。原代码 `n = number - 1` 的 off-by-one 就这样，
    /// 权重是在这套「错位」特征上训练的：C 用 N 的行、N 用 O 的行、O 用 F 的行。
    /// 注意 F（Z=9）在 legacy 下越界报错，这与原实现的 `IndexError` 行为一致。
    pub fn row_for(&self, sym: &str, mode: IndexMode) -> Result<usize, String> {
        let z = atomic_number(sym).ok_or_else(|| format!("不支持的原子类型: {sym}"))?;
        match mode {
            IndexMode::Corrected => SYMBOLS
                .iter()
                .position(|s| *s == sym)
                .ok_or_else(|| format!("元素 {sym} 不在固化的 8 元素表内")),
            IndexMode::Legacy => {
                let row = z - 1;
                if row < self.rows {
                    Ok(row)
                } else {
                    Err(format!(
                        "元素 {sym}（Z={z}）超出原实现的索引范围（复现 n=Z-1，只支持 Z<=8 的 H/Li/Be/B/C/N/O）"
                    ))
                }
            }
        }
    }

    /// 按 `syms` 组装模型输入 `x_geom`（N × 49）。
    pub fn build_x_geom(&self, syms: &[String], mode: IndexMode) -> Result<Vec<f32>, String> {
        let mut out = Vec::with_capacity(syms.len() * self.cols);
        for s in syms {
            let r = self.row_for(s, mode)?;
            out.extend_from_slice(&self.table[r * self.cols..(r + 1) * self.cols]);
        }
        Ok(out)
    }
}
