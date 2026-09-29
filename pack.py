"""打包 wmview.boom 插件压缩包。

产物: wmview.boom.zip（文件名不带版本号，版本号见包内 info.json）
解压到主程序根目录 plugs/ 下（目录名 wmview.boom），重启 wmview 即可加载。

包内容 = 前端 MF 产物（index.js + assets/）。assets/ 里已经包含推理所需的全部资源：
Rust→wasm 推理核（约 280 KB）与两个模型的扁平权重（各 14.5 MB）。
**不含** python 运行时（旧版内置的 python-3.10 已随 Rust 移植删除），
也**不含** 模型/ 目录（原项目源码与 .pt 权重只在开发侧使用）。

构建顺序：`npm run build`（type-check → wasm-pack 出 boom-core/pkg → vite 出 dist），
然后再跑本脚本。cargo / wasm-pack 只在本机打包时需要，客户端不需要。

用法: python pack.py
"""

import json
import os
from zipfile import ZIP_DEFLATED, ZipFile

# 运行期不需要的东西不打包
SKIP_DIRS = {'__pycache__'}
SKIP_SUFFIX = ('.pyc',)


def compress_specified_items(output_zip: str, items: list, compress_level: int = 6):
    with ZipFile(output_zip, "w", ZIP_DEFLATED, compresslevel=compress_level) as zipf:
        for item in items:
            if not os.path.exists(item):
                print(f"警告：{item} 不存在，已跳过")
                continue
            if os.path.isfile(item):
                zipf.write(item, arcname=os.path.basename(item))
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    for file in files:
                        if file.endswith(SKIP_SUFFIX):
                            continue
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(item))
                        zipf.write(file_path, arcname=arcname)


# 版本号单一来源：package.json 的 version，打包时同步进 info.json 随包发布
with open("package.json", encoding="utf-8") as f:
    VERSION = json.load(f)["version"]

with open("info.json", encoding="utf-8") as f:
    _info = json.load(f)
if _info.get("version") != VERSION:
    _info["version"] = VERSION
    with open("info.json", "w", encoding="utf-8") as f:
        json.dump(_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"已同步 info.json 的 version → {VERSION}")

# ====================== 【只需修改这里】 ======================
TO_COMPRESS = [
    "dist/index.js",   # MF remote 入口
    "dist/assets",     # 运行时 chunk + css + wasm + 扁平权重（inference core）
    "info.json",       # 插件市场元数据（含版本号）
    "src/icon.png",    # 插件图标 → zip 根目录 icon.png
]

OUTPUT = "wmview.boom.zip"
compress_specified_items(OUTPUT, TO_COMPRESS, compress_level=6)
size = os.path.getsize(OUTPUT) / 1e6
print(f"压缩完成: {OUTPUT} ({size:.1f} MB)")
