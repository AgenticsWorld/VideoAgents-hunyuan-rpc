#!/usr/bin/env python3
"""Stage D:产物打包——collider.glb、thumbnail.jpg、world_pano.jpg、artifacts.json 清单。

- collider.glb:合并分层 mesh(轴向修正同 stage_b)→ open3d 减面(默认 65536 三角形)→ trimesh 导 GLB;
  只作视窗碰撞体,不带纹理。
- thumbnail.jpg:复用 stage_b 的中心视图渲染(view_000.png),转 JPEG。
- world_pano.jpg:输入全景转 JPEG(视窗「全景图」预览与排障用)。
- artifacts.json:客户端(modules/hunyuan_world.py)要下载的全部产物清单 + 标定元数据(gpo_raw 等)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def axis_fix(vertices: np.ndarray) -> np.ndarray:
    return np.stack([vertices[:, 1], -vertices[:, 2], -vertices[:, 0]], axis=1)


def build_collider(job: Path, layers: list[str], target_tris: int, log=print) -> str | None:
    import open3d as o3d
    import trimesh

    verts_all, faces_all = [], []
    offset = 0
    for name in layers:
        m = o3d.io.read_triangle_mesh(str(job / 'scenegen' / name))
        if not m.has_vertices():
            continue
        v = axis_fix(np.asarray(m.vertices))
        f = np.asarray(m.triangles)
        verts_all.append(v)
        faces_all.append(f + offset)
        offset += len(v)
    if not verts_all:
        log('[stage_d] 没有 mesh 可合并,collider 跳过')
        return None
    merged = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(verts_all)),
                                       o3d.utility.Vector3iVector(np.concatenate(faces_all)))
    n0 = len(merged.triangles)
    if n0 > target_tris:
        merged = merged.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    merged.remove_unreferenced_vertices()
    tm = trimesh.Trimesh(vertices=np.asarray(merged.vertices), faces=np.asarray(merged.triangles), process=False)
    tm.export(job / 'collider.glb')
    log(f'[stage_d] collider.glb:{n0} → {len(merged.triangles)} 三角形')
    return 'collider.glb'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--collider-tris', type=int, default=65536)
    args = ap.parse_args()

    job = Path(args.job_dir)
    from PIL import Image

    rec_a = json.loads((job / 'stage_a.json').read_text(encoding='utf-8'))
    rec_c = json.loads((job / 'stage_c.json').read_text(encoding='utf-8'))
    artifacts: dict = {'splats': {res: f'gs/{name}' for res, name in rec_c['splats'].items()}}   # 文件在 gs/ 子目录

    col = build_collider(job, rec_a['layers'], args.collider_tris)
    if col:
        artifacts['collider'] = col

    center = job / 'renders' / 'view_000.png'
    if center.is_file():
        Image.open(center).convert('RGB').save(job / 'thumbnail.jpg', quality=90)
        artifacts['thumbnail'] = 'thumbnail.jpg'
    if (job / 'pano.png').is_file():
        Image.open(job / 'pano.png').convert('RGB').save(job / 'world_pano.jpg', quality=92)
        artifacts['world_pano'] = 'world_pano.jpg'

    meta = {'gpo_raw': rec_c.get('gpo_raw'), 'steps': rec_c.get('steps'), 'n_final': rec_c.get('n_final'),
            'classes': rec_a.get('classes'), 'seconds': {'scenegen': rec_a.get('seconds'), 'gs': rec_c.get('seconds')}}
    (job / 'pack_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    artifacts['pack_meta'] = 'pack_meta.json'

    (job / 'artifacts.json').write_text(json.dumps({'artifacts': artifacts, 'meta': meta}, ensure_ascii=False, indent=1),
                                        encoding='utf-8')
    print(f'[stage_d] 完成:{json.dumps(artifacts, ensure_ascii=False)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
