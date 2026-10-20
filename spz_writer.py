"""SPZ v2 编码器(纯 Python,零原生依赖):3DGS 参数 → Spark / spz 加载器可读的 .spz。

格式依据 nianticlabs/spz src/cc/load-spz.cc(version 2,gzip 封装):
  文件 = gzip(序列化缓冲)。
  头 16 B:magic u32 LE = 0x5053474e("NGSP")、version u32 = 2、numPoints u32、
          shDegree u8、fractionalBits u8 = 12、flags u8(bit0 = antialiased)、reserved u8 = 0。
  体(planar,逐属性整段,点序一致;**段落顺序必须与官方一致**——serializePackedGaussians 的顺序是
  positions → alphas → colors → scales → rotations → sh,与 checkSizes 的检查顺序不同,
  踩过坑:按错顺序写,位置正常但颜色/尺度/旋转全错位,渲染糊成一片):
    positions N*3*3 B:24 bit 有符号定点(LE,缩放 2^12);
    alphas    N B    :uint8(sigmoid(logit) * 255)       —— 输入是 logit(未过 sigmoid 的不透明度);
    colors    N*3 B  :uint8(dc * (0.15 * 255) + 127.5)  —— 输入是 SH DC 系数(rgb = dc * SH_C0 + 0.5);
    scales    N*3 B  :uint8((log_scale + 10) * 16)      —— 输入是 log(尺度);
    rotations N*3 B  :v2 "first-three":单位四元数(xyzw 序)把 w 转正,存 xyz:uint8(q * ±127.5 + 127.5);
    sh        本编码器只出 shDegree = 0(DC-only),无 SH 段。

只依赖 numpy。写文件用 write_spz();内存取用 encode_spz()。
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x5053474E          # "NGSP"
VERSION = 2
FRACTIONAL_BITS = 12
COLOR_SCALE = 0.15
FLAG_ANTIALIASED = 0x1

_POS_SCALE = float(1 << FRACTIONAL_BITS)
_POS_MIN, _POS_MAX = -(1 << 23), (1 << 23) - 1


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _pack_positions(means: np.ndarray) -> bytes:
    fixed = np.clip(np.round(means.astype(np.float64) * _POS_SCALE), _POS_MIN, _POS_MAX).astype(np.int32)
    # 24 bit 小端(二进制补码低 24 位;加载端做符号扩展)
    u = (fixed & 0xFFFFFF).astype(np.uint32)
    out = np.empty((fixed.shape[0], 3, 3), dtype=np.uint8)
    out[:, :, 0] = (u & 0xFF).astype(np.uint8)
    out[:, :, 1] = ((u >> 8) & 0xFF).astype(np.uint8)
    out[:, :, 2] = ((u >> 16) & 0xFF).astype(np.uint8)
    return out.tobytes()


def _pack_scales(log_scales: np.ndarray) -> bytes:
    return _u8((log_scales + 10.0) * 16.0).tobytes()


def _pack_rotations_first_three(quats_xyzw: np.ndarray) -> bytes:
    q = quats_xyzw.astype(np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    q = q / norm
    neg = q[:, 3] < 0                       # w 取正(-q 与 q 同旋转)
    q[neg] = -q[neg]
    return _u8(q[:, :3] * 127.5 + 127.5).tobytes()


def _pack_alphas(logits: np.ndarray) -> bytes:
    sig = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return _u8(sig * 255.0).tobytes()


def _pack_colors(dc_colors: np.ndarray) -> bytes:
    return _u8(dc_colors * (COLOR_SCALE * 255.0) + 127.5).tobytes()


def encode_spz(means: np.ndarray, quats_xyzw: np.ndarray, log_scales: np.ndarray,
               logits: np.ndarray, dc_colors: np.ndarray, *, antialiased: bool = False) -> bytes:
    """N 个高斯 → .spz 文件字节。数组形状:means/quats/log_scales/dc_colors (N,3)/(N,4)/(N,3)/(N,3),logits (N,)。"""
    n = int(means.shape[0])
    for name, arr, cols in (('means', means, 3), ('quats', quats_xyzw, 4), ('log_scales', log_scales, 3),
                            ('dc_colors', dc_colors, 3)):
        if arr.shape != (n, cols):
            raise ValueError(f'{name} 形状须为 ({n}, {cols}),得 {arr.shape}')
    if logits.shape != (n,):
        raise ValueError(f'logits 形状须为 ({n},),得 {logits.shape}')
    header = struct.pack('<III4B', MAGIC, VERSION, n, 0, FRACTIONAL_BITS,
                         FLAG_ANTIALIASED if antialiased else 0, 0)
    body = b''.join((_pack_positions(means), _pack_alphas(logits), _pack_colors(dc_colors),
                     _pack_scales(log_scales), _pack_rotations_first_three(quats_xyzw)))
    return gzip.compress(header + body, compresslevel=6)


def write_spz(path: str | Path, means: np.ndarray, quats_xyzw: np.ndarray, log_scales: np.ndarray,
              logits: np.ndarray, dc_colors: np.ndarray, *, antialiased: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_spz(means, quats_xyzw, log_scales, logits, dc_colors, antialiased=antialiased))
    return path
