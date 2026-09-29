//! 扁平权重：`<target>.json`（张量索引）+ `<target>.bin`（f32 小端，行主序）。
//!
//! 与 `tools/export_flat_weights.py` 的产物一一对应。之所以不用 npz：npz 是 zip
//! 容器，wasm 里解压不划算；扁平格式一次 fetch 就能直接当成 f32 切片用。
//! （与 wmvie.flavor 的 flavor-core/src/weights.rs 同一理由。）

use std::collections::HashMap;

/// 单个张量在二进制里的位置与形状（形状按行主序，offset 以 **元素个数** 计）。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub offset: usize,
}

/// 权重索引文件（字段比 flavor 多：带上模型元信息，前端直接用）。
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WeightIndex {
    #[serde(default)]
    pub target: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub unit: String,
    #[serde(default)]
    pub epoch: Option<i64>,
    #[serde(default)]
    pub val_mae: Option<f64>,
    #[serde(default)]
    pub n_param: Option<usize>,
    #[serde(default)]
    pub dtype: String,
    pub tensors: HashMap<String, TensorInfo>,
}

/// 只读权重表：一份 f32 数据 + 名字 -> (shape, offset) 的索引。
pub struct Weights {
    data: Vec<f32>,
    index: WeightIndex,
}

impl Weights {
    /// 由索引 JSON 文本 + f32 小端字节流构造（构造时校验所有张量不越界）。
    pub fn from_parts(index_json: &str, bin: &[u8]) -> Result<Self, String> {
        let index: WeightIndex =
            serde_json::from_str(index_json).map_err(|e| format!("权重索引解析失败：{e}"))?;
        if bin.len() % 4 != 0 {
            return Err(format!("权重二进制长度 {} 不是 4 的倍数", bin.len()));
        }
        let data: Vec<f32> = bin
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (name, t) in &index.tensors {
            let n: usize = t.shape.iter().product();
            if t.offset + n > data.len() {
                return Err(format!(
                    "张量 {name} 越界：offset {} + {n} > {}",
                    t.offset,
                    data.len()
                ));
            }
        }
        Ok(Self { data, index })
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.index.tensors.len()
    }

    pub fn meta(&self) -> &WeightIndex {
        &self.index
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.tensors.contains_key(name)
    }

    /// 张量形状（缺失即 panic：调用前应已由 `Model::new` 做过结构自检）。
    pub fn shape(&self, name: &str) -> &[usize] {
        match self.index.tensors.get(name) {
            Some(t) => &t.shape,
            None => panic!("缺少权重张量 {name}"),
        }
    }

    /// 取张量的连续 f32 切片（行主序）。
    pub fn get(&self, name: &str) -> &[f32] {
        match self.index.tensors.get(name) {
            Some(t) => {
                let n: usize = t.shape.iter().product();
                &self.data[t.offset..t.offset + n]
            }
            None => panic!("缺少权重张量 {name}"),
        }
    }

    /// `nn.Sequential` 的子模块下标：名字形如 `{prefix}.{i}.weight` 的 i（升序去重）。
    ///
    /// 对应 `runtime/grmp_numpy.py:_child_indices`——层数、MLP 宽度、融合方式全部由权重
    /// 驱动，所以换 checkpoint 不用改代码。
    pub fn child_indices(&self, prefix: &str) -> Vec<usize> {
        let head = format!("{prefix}.");
        let mut out: Vec<usize> = self
            .index
            .tensors
            .keys()
            .filter_map(|k| {
                let rest = k.strip_prefix(&head)?;
                if !k.ends_with(".weight") {
                    return None;
                }
                rest.split('.').next()?.parse::<usize>().ok()
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// 结构自检：`names` 里每个张量都必须存在（缺失时给出一条完整报告）。
    pub fn require(&self, names: &[&str]) -> Result<(), String> {
        let missing: Vec<&str> = names.iter().copied().filter(|n| !self.has(n)).collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "权重缺少 {} 个张量：{}（checkpoint 结构可能与本实现不兼容）",
                missing.len(),
                missing.join(", ")
            ))
        }
    }

    /// 按名字典序返回全部张量名（自检/诊断用）。
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.tensors.keys().cloned().collect();
        v.sort();
        v
    }
}
