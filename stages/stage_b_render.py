#!/usr/bin/env python3
"""Stage B:分层 mesh → 有界轨迹多视图渲染(Open3D OffscreenRenderer,Windows 可用)。

输入 <job>/scenegen/mesh_layer*.ply(1.0 世界系:z 向上、全景中心列 = -x,
见 hy3dworld/utils/general_utils.py spherical_uv_to_directions)。
统一换到「marble raw」契约系(x 右、y 下、+z = 全景中心列,与 world.json#alignment 配套):
  v' = (y, -z, -x)   # 正交旋转,det = +1,三角面绕序不变
MoGe 深度是相对尺度:以「相机下方地面平面拟合」估 gpo_raw(原始单位的相机离地高),
world.json 的 scale_fix = 已知相机高(camera_height_m,白模精确值)/ gpo_raw,同 Marble 标定口径。

轨迹(严格限制在全景包络内,远离 1.0 几何的边缘退化区):
  1 个中心视图(朝全景中心列)+ 绕竖直轴半径 radius_raw 的环上 N-1 个视图,
  内外朝向交替( inward 补中心视差,outward 补墙面);高差 0(保守)。

产物:
  <job>/renders/view_XXX.png / view_XXX_depth.npy(原始单位,相机系 z 深度)
  <job>/renders/cameras.json  {views: [{file, depth, K, c2w}], gpo_raw, radius_raw, frame}
深度与相机同一套原始单位,Stage C 训练与 SPZ 导出全程不引入米制,标定只发生在 world.json。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def axis_fix(vertices: np.ndarray) -> np.ndarray:
    """1.0 世界系(z 上、中心列=-x)→ marble raw(x 右、y 下、z=中心列):v'=(y,-z,-x)。"""
    return np.stack([vertices[:, 1], -vertices[:, 2], -vertices[:, 0]], axis=1)


def load_merged_mesh(scenegen: Path, layers: list[str]):
    import open3d as o3d
    meshes = []
    for name in layers:
        m = o3d.io.read_triangle_mesh(str(scenegen / name))
        if not m.has_vertices():
            print(f'[stage_b] 警告:{name} 是空 mesh,跳过', flush=True)
            continue
        m.vertices = o3d.utility.Vector3dVector(axis_fix(np.asarray(m.vertices)))
        # 非有限顶点清洗(2026-10-20):MoGe 极点深度常出 inf/nan,open3d/Filament 会把整个 mesh 丢弃,
        # 表现为渲染全黑、深度全 inf
        v = np.asarray(m.vertices)
        finite = np.isfinite(v).all(axis=1)
        if not finite.all():
            print(f'[stage_b] {name}:剔除非有限顶点 {(~finite).sum()}/{len(v)}', flush=True)
            m = m.select_by_index(np.flatnonzero(finite).tolist())
        if not m.has_vertex_normals():
            m.compute_vertex_normals()
        vv = np.asarray(m.vertices)
        print(f'[stage_b] {name}:顶点 {len(vv)},三角 {len(m.triangles)},'
              f'bounds x[{vv[:, 0].min():.2f},{vv[:, 0].max():.2f}] y[{vv[:, 1].min():.2f},{vv[:, 1].max():.2f}] '
              f'z[{vv[:, 2].min():.2f},{vv[:, 2].max():.2f}]', flush=True)
        # 绕序翻转(2026-10-20):1.0 的查看器双面渲染,官方不保证绕序朝内;open3d/Filament 默认背面剔除,
        # 从房间内部看全部剔除 → 渲染全黑、深度全 inf。交换两列即反转绕序
        m.triangles = o3d.utility.Vector3iVector(np.asarray(m.triangles)[:, [0, 2, 1]])
        meshes.append(m)
    if not meshes:
        raise SystemExit('[stage_b] 没有可用分层 mesh')
    return meshes


def fit_ground_offset(meshes) -> float:
    """估相机(原点)到脚下地面的距离(原始单位,y 向下为正):取原点附近 y>0 顶点的
    水平距离近邻集,众数化(y 直方图最高峰)。天空层顶点被距离门限天然排除。"""
    ys, hs = [], []
    for m in meshes:
        v = np.asarray(m.vertices)
        y = v[:, 1]
        horiz = np.hypot(v[:, 0], v[:, 2])
        ys.append(y)
        hs.append(horiz)
    y = np.concatenate(ys)
    h = np.concatenate(hs)
    pos = y > 0
    if pos.sum() < 100:
        return 1.0                       # 拟合不了就 1.0,标定全交给 scale_fix
    med = float(np.median(y[pos]))
    near = pos & (h < 1.5 * med)
    if near.sum() < 100:
        return med
    hist, edges = np.histogram(y[near], bins=60, range=(0, np.percentile(y[near], 99)))
    peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    return float(max(peak, 1e-3))


def trajectory(n_views: int, radius: float) -> list[dict]:
    """中心 1 视图 + 环上 N-1 视图(内外交替)。位姿为 c2w(4x4,marble raw 系,y 下 z 前)。"""
    views = []

    def look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        fwd = target - eye
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        down = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, down)
        n = np.linalg.norm(right)
        right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        true_down = np.cross(fwd, right)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, true_down, fwd, eye
        return m

    views.append({'eye': np.zeros(3), 'c2w': look_at_c2w(np.zeros(3), np.array([0.0, 0.0, 1.0])), 'kind': 'center'})
    ring = max(n_views - 1, 4)
    for i in range(ring):
        a = 2 * math.pi * i / ring
        eye = np.array([radius * math.cos(a), 0.0, radius * math.sin(a)])
        inward = (i % 2 == 0)
        target = np.zeros(3) if inward else eye * 2.0        # inward 看轴心,outward 沿径向看墙
        views.append({'eye': eye, 'c2w': look_at_c2w(eye, target), 'kind': 'ring_in' if inward else 'ring_out'})
    return views


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--views', type=int, default=24)
    ap.add_argument('--camera-height-m', type=float, default=1.6, help='全景相机真实离地高(白模精确值)')
    ap.add_argument('--radius-m', type=float, default=0.35, help='轨迹环半径(米,按标定换算进原始单位)')
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=576)
    ap.add_argument('--fov-h-deg', type=float, default=90.0)
    args = ap.parse_args()

    job = Path(args.job_dir)
    rec_a = json.loads((job / 'stage_a.json').read_text(encoding='utf-8'))
    scenegen = job / 'scenegen'
    out = job / 'renders'
    out.mkdir(exist_ok=True)

    import open3d as o3d
    from PIL import Image

    meshes = load_merged_mesh(scenegen, rec_a['layers'])
    gpo_raw = fit_ground_offset(meshes)
    radius_raw = args.radius_m * (gpo_raw / max(args.camera_height_m, 0.1))
    print(f'[stage_b] gpo_raw={gpo_raw:.4f}(原始单位相机高),环半径 {radius_raw:.4f}', flush=True)

    # CPU 光线投射(2026-10-20):本机 OffscreenRenderer/Filament-Vulkan 实机验证不可用(最小立方体也全黑),
    # 换 open3d RaycastingScene(Embree,无 GL 依赖);RGB 用重心坐标插值顶点色,射线 dir_z=1 ⇒ t 即 z 深度
    import open3d.core as o3c
    from PIL import Image

    scene = o3d.t.geometry.RaycastingScene()
    color_layers = []                             # (tri_offset, vertex_colors, triangles)
    tri_offset = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float32)
        t = np.asarray(m.triangles, dtype=np.uint32)
        scene.add_triangles(o3c.Tensor(v), o3c.Tensor(t))
        cols = (np.asarray(m.vertex_colors, dtype=np.float32) if m.has_vertex_colors()
                else np.full((len(v), 3), 0.7, dtype=np.float32))
        color_layers.append((tri_offset, cols, t))
        tri_offset += len(t)

    W, H = args.width, args.height
    fx = 0.5 * W / math.tan(math.radians(args.fov_h_deg) / 2)
    K = [fx, 0.0, W / 2.0, 0.0, fx, H / 2.0, 0.0, 0.0, 1.0]
    px, py = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
    cam_dirs = np.stack([(px - W / 2.0) / fx, (py - H / 2.0) / fx, np.ones_like(px)], axis=-1)
    NO_HIT = 4294967295

    views = trajectory(args.views, radius_raw)
    records = []
    for i, v in enumerate(views):
        c2w = v['c2w']
        R = c2w[:3, :3].astype(np.float32)
        eye = c2w[:3, 3].astype(np.float32)
        dirs = (cam_dirs @ R.T).reshape(-1, 3).astype(np.float32)
        rays = np.concatenate([np.broadcast_to(eye, dirs.shape).copy(), dirs], axis=1)
        ans = scene.cast_rays(o3c.Tensor(rays))
        t_hit = ans['t_hit'].numpy()
        pid = ans['primitive_ids'].numpy()
        puv = ans['primitive_uvs'].numpy()
        hit = pid != NO_HIT
        img = np.zeros((H * W, 3), dtype=np.float32)
        for tri_off, cols, tris in color_layers:
            sel = hit & (pid >= tri_off) & (pid < tri_off + len(tris))
            if not sel.any():
                continue
            tri = tris[pid[sel] - tri_off]
            w0, w1 = puv[sel, 0], puv[sel, 1]
            img[sel] = ((1 - w0 - w1)[:, None] * cols[tri[:, 0]]
                        + w0[:, None] * cols[tri[:, 1]] + w1[:, None] * cols[tri[:, 2]])
        depth = t_hit.reshape(H, W).astype(np.float32)
        depth[~hit.reshape(H, W)] = np.inf
        rgb8 = (np.clip(img.reshape(H, W, 3), 0, 1) * 255).astype(np.uint8)
        name = f'view_{i:03d}.png'
        Image.fromarray(rgb8).save(out / name)
        np.save(out / f'view_{i:03d}_depth.npy', depth)
        finite = np.isfinite(depth)
        records.append({'file': name, 'depth': f'view_{i:03d}_depth.npy', 'K': [round(x, 6) for x in K],
                        'c2w': [[round(x, 8) for x in row] for row in c2w.tolist()], 'kind': v['kind']})
        print(f'[stage_b] {i + 1}/{len(views)} {name}({v["kind"]}) '
              f'深度有效 {finite.mean():.0%},中位 {float(np.median(depth[finite])) if finite.any() else -1:.3f}', flush=True)

    meta = {'views': records, 'width': args.width, 'height': args.height, 'fov_h_deg': args.fov_h_deg,
            'gpo_raw': gpo_raw, 'radius_raw': radius_raw, 'frame': 'marble_raw(x 右,y 下,+z=全景中心列)'}
    (out / 'cameras.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[stage_b] 完成:{len(records)} 视图 → {out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
