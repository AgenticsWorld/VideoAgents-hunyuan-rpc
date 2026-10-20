"""SPZ v2 编码器(gpu_worker/spz_writer.py)自检:头部布局、各属性段尺寸、量化往返误差。

不依赖 torch/GPU;对照 nianticlabs/spz load-spz.cc(version 2,gzip 封装)的读取口径逐项解码验证。
"""
import gzip
import pytest
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spz_writer import COLOR_SCALE, FRACTIONAL_BITS, MAGIC, VERSION, encode_spz  # noqa: E402


def make_gaussians(n=5, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 2, (n, 3)).astype(np.float32)
    quats = rng.normal(0, 1, (n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    log_scales = rng.uniform(-9, 1, (n, 3)).astype(np.float32)
    logits = rng.normal(0, 2, n).astype(np.float32)
    dc = rng.normal(0, 1, (n, 3)).astype(np.float32)
    return means, quats, log_scales, logits, dc


def decode(buf, n):
    """按 SPZ v2 官方布局(positions → alphas → colors → scales → rotations)解出各段。"""
    raw = gzip.decompress(buf)
    magic, version, num, sh, frac, flags, reserved = struct.unpack('<III4B', raw[:16])
    off = 16
    pos = np.frombuffer(raw, dtype=np.uint8, count=n * 9, offset=off).reshape(n, 3, 3)
    off += n * 9
    alphas = np.frombuffer(raw, dtype=np.uint8, count=n, offset=off).astype(np.float64)
    off += n
    colors = np.frombuffer(raw, dtype=np.uint8, count=n * 3, offset=off).astype(np.float64)
    off += n * 3
    scales = np.frombuffer(raw, dtype=np.uint8, count=n * 3, offset=off).astype(np.float64) / 16.0 - 10.0
    off += n * 3
    rots = np.frombuffer(raw, dtype=np.uint8, count=n * 3, offset=off).astype(np.float64)
    off += n * 3
    assert off == len(raw), '各段读完应恰好到缓冲末尾(shDegree=0 无 SH 段)'
    return (magic, version, num, sh, frac, flags, reserved), pos, scales, rots, alphas, colors


def unpack_positions(packed):
    u = (packed[:, :, 0].astype(np.int32) | (packed[:, :, 1].astype(np.int32) << 8)
         | (packed[:, :, 2].astype(np.int32) << 16))
    u = np.where(u & 0x800000, u - (1 << 24), u)          # 24 位符号扩展
    return u.astype(np.float64) / (1 << FRACTIONAL_BITS)


def test_header_layout():
    means, quats, ls, lo, dc = make_gaussians()
    hdr, *_ = decode(encode_spz(means, quats, ls, lo, dc, antialiased=True), len(means))
    magic, version, num, sh, frac, flags, reserved = hdr
    assert (magic, version, num, sh, frac, reserved) == (MAGIC, VERSION, len(means), 0, FRACTIONAL_BITS, 0)
    assert flags & 0x1, 'antialiased 应置 flags bit0'


def test_positions_roundtrip_within_quantization():
    means, quats, ls, lo, dc = make_gaussians(n=200)
    _, pos, *_ = decode(encode_spz(means, quats, ls, lo, dc), len(means))
    got = unpack_positions(pos)
    assert np.abs(got - means).max() <= 0.5 / (1 << FRACTIONAL_BITS) + 1e-6


def test_scales_alphas_colors_roundtrip():
    n = 200
    means, quats, ls, lo, dc = make_gaussians(n=n)
    _, _, scales, _, alphas, colors = decode(encode_spz(means, quats, ls, lo, dc), n)
    assert np.abs(scales.reshape(n, 3) - ls).max() <= 1 / 16 + 1e-6         # 4 bit 有效精度
    sig = 1 / (1 + np.exp(-lo))
    assert np.abs(alphas / 255.0 - sig).max() <= 1 / 255 + 1e-6
    assert np.abs((colors.reshape(n, 3) - 127.5) / (COLOR_SCALE * 255) - dc).max() <= 1 / (COLOR_SCALE * 255) + 1e-6


def test_rotation_w_made_positive_and_xyz_stored():
    quats = np.array([[0.0, 0.0, 0.0, -1.0], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, 0.5, 0.5]], dtype=np.float32)
    n = len(quats)
    buf = encode_spz(np.zeros((n, 3), np.float32), quats, np.zeros((n, 3), np.float32),
                     np.zeros(n, np.float32), np.zeros((n, 3), np.float32))
    _, _, _, rots, _, _ = decode(buf, n)
    expect = quats.copy()
    expect[expect[:, 3] < 0] *= -1                                          # w 转正,-q 同旋转
    assert np.allclose(rots.reshape(n, 3), expect[:, :3] * 127.5 + 127.5, atol=1.0)


def test_shape_validation():
    means, quats, ls, lo, dc = make_gaussians()
    with pytest.raises(ValueError):
        encode_spz(means[:3], quats, ls, lo, dc)          # means 数量对不上
    with pytest.raises(ValueError):
        encode_spz(means, quats[:, :3], ls, lo, dc)       # quats 列数不对
    with pytest.raises(ValueError):
        encode_spz(means, quats, ls, lo[:3], dc)          # logits 数量对不上
