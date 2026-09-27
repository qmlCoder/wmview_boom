"""生成 src/icon.png（纯标准库，不依赖 PIL）。

图标：深色圆角底 + 橙/黄色放射状爆轰纹 + 中心高光，尺寸 128×128。
用法: python tools/make_icon.py
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "icon.png"
SIZE = 128


def lerp(a, b, t):
    return a + (b - a) * t


def ray_alpha(angle, seed=0.0):
    """24 条放射纹：返回 0~1 的强度。"""
    n = 12
    phase = (angle * n / (2 * math.pi) + seed) % 1.0
    return max(0.0, math.cos((phase - 0.5) * 2 * math.pi * 0.5) ** 2)


def pixel(x, y):
    cx = cy = (SIZE - 1) / 2
    dx, dy = x - cx, y - cy
    r = math.hypot(dx, dy) / (SIZE / 2)
    ang = math.atan2(dy, dx)
    # 圆角方形遮罩
    corner = 0.22
    qx = abs(dx) / (SIZE / 2 - 6)
    qy = abs(dy) / (SIZE / 2 - 6)
    outside = max(qx, qy) > 1.0 or (qx > 1 - corner and qy > 1 - corner
                                    and math.hypot(qx - (1 - corner), qy - (1 - corner)) > corner)
    if outside:
        return (0, 0, 0, 0)

    # 底色：深蓝 → 深紫
    t = 0.5 * (1 + math.sin(ang))
    base = (int(lerp(18, 46, t)), int(lerp(22, 26, t)), int(lerp(46, 66, t)))

    # 放射纹
    a = ray_alpha(ang) * max(0.0, 1.0 - r) ** 1.4
    hot = min(1.0, a * 1.6 + max(0.0, 0.42 - r) * 1.4)
    col = (
        min(255, int(base[0] + hot * (255 - base[0]))),
        min(255, int(base[1] + hot * (190 - base[1]))),
        min(255, int(base[2] + hot * (60 - base[2]))),
    )
    # 中心高光
    if r < 0.16:
        w = 1 - r / 0.16
        col = (min(255, int(col[0] + w * 60)), min(255, int(col[1] + w * 70)),
               min(255, int(col[2] + w * 90)))
    return (col[0], col[1], col[2], 255)


def main():
    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)  # filter type 0
        for x in range(SIZE):
            raw.extend(pixel(x, y))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
           + chunk(b'IEND', b''))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(png)
    print(f'written: {OUT} ({len(png)} bytes)')


if __name__ == '__main__':
    main()
