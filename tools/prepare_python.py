"""准备插件自带的 python 运行时（Windows，仅打包/联调用）。

做的事：
1. 从 ../python-3.10 拷贝一份嵌入式 Python 3.10 到插件目录；
2. 往里面装 numpy（推理运行时唯一的第三方依赖）。

推理运行时（runtime/）只用 numpy：模型前向、几何特征、基函数全是 numpy 实现，
不需要 torch / rdkit / torch_geometric（见 README 的「运行时方案」一节）。

用法：
    python tools/prepare_python.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "python-3.10"
DST = ROOT / "python-3.10"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple/"
REQUIRED = ["numpy"]

# 内置运行时只用来跑 runtime/：pip/setuptools/ssl/sqlite 这些开发期组件用不上，
# 删掉能省 ~10MB 打包体积（prune 后必须能通过 tools/check_golden.py）
PRUNE = [
    "get-pip.py",
    "Lib/site-packages/pip",
    "Lib/site-packages/setuptools",
    "Lib/site-packages/wheel",
    "Lib/site-packages/_distutils_hack",
    "Lib/site-packages/distutils-precedence.pth",
    "Scripts",
]


def copy_runtime(recopy: bool = False):
    if not SRC.exists():
        raise SystemExit(f"找不到嵌入式 python: {SRC}")
    if DST.exists():
        if recopy:
            # 不直接删：改名成 .bak 留一份可回退的副本
            bak = DST.with_name(DST.name + ".bak")
            if bak.exists():
                shutil.rmtree(bak)
            DST.rename(bak)
            print(f"已把旧的运行时改名为 {bak.name}，重新拷贝一份")
            shutil.copytree(SRC, DST)
            print(f"已拷贝 python 运行时: {SRC} -> {DST}")
            return
        print(f"已存在，跳过拷贝: {DST}")
        return
    shutil.copytree(SRC, DST)
    print(f"已拷贝 python 运行时: {SRC} -> {DST}")


def install_deps():
    py = DST / "python.exe"
    if not py.exists():
        raise SystemExit(f"找不到 {py}")
    for pkg in REQUIRED:
        subprocess.check_call([str(py), "-m", "pip", "install", pkg, "-i", MIRROR,
                               "--disable-pip-version-check", "-q"])
        print(f"已安装 {pkg}")
    subprocess.check_call([str(py), "-c", "import numpy;print('numpy', numpy.__version__)"])


def prune():
    for rel in PRUNE:
        target = DST / rel
        if target.is_dir():
            shutil.rmtree(target)
            print(f"已删除目录 {rel}")
        elif target.exists():
            target.unlink()
            print(f"已删除文件 {rel}")
    # 同时清掉 numpy 的 dist-info 里用不到的东西：留着也无妨，这里只删 *.dist-info 之外的空目录
    for info in (DST / "Lib" / "site-packages").glob("pip-*.dist-info"):
        shutil.rmtree(info, ignore_errors=True)
    for info in (DST / "Lib" / "site-packages").glob("setuptools-*.dist-info"):
        shutil.rmtree(info, ignore_errors=True)
    for info in (DST / "Lib" / "site-packages").glob("wheel-*.dist-info"):
        shutil.rmtree(info, ignore_errors=True)


def main():
    recopy = "--recopy" in sys.argv
    copy_runtime(recopy)
    install_deps()
    prune()
    subprocess.check_call([str(DST / "python.exe"), "-c", "import numpy;print('after prune, numpy', numpy.__version__)"])
    print("完成。联调时前端会调用", (DST / 'pythonw.exe').as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
