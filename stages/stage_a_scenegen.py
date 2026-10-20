#!/usr/bin/env python3
"""Stage A:全景 → HunyuanWorld-1.0 分层 3D 重建(demo_scenegen.py 子进程)。

在 hyworld conda env 里跑(依赖 HunyuanWorld-1.0 仓库及其环境);产物:
  <job>/scenegen/mesh_layer*.ply   分层 textured mesh(z 向上、全景中心列 = -x 的世界系)
  <job>/stage_a.json               {layers: [...], classes}
worker 调用约定:--job-dir 下有 pano.png;本脚本幂等,重跑覆盖 scenegen/。

1.0 官方仓库的补丁由 ensure_patches() 在每次运行前幂等应用(2026-10-19,Windows 3090 实机踩坑记录,
详见 docs/hunyuan_deploy.md「必打补丁」):
  P1 inpaint_utils.py   enable_model_cpu_offload → enable_sequential_cpu_offload
                        (整模型卸载会把 24GB bf16 transformer 整体搬上 GPU;逐层卸载峰值仅几 GB,
                        bf16 直跑,绕开 AngelSlim FP8——后者在 Windows torch 2.5 必现 access violation)
  P2 layer_decomposer.py build_sr_model 加 tile=512(RealESRGAN 大图不分块必 OOM,且 OOM 被
                        sr_inference 吞掉只打印,到天空阶段才报 FileNotFoundError)
  P3 sr_utils.py        小图(width≤1536)关 tile(tile_process 在 Windows 小图上原生 stack overflow)
  P4 layer_utils.py     _clear_models 容错:顺序卸载下参数在 meta 设备,pipeline.to('cpu') 必抛
                        NotImplementedError(Cannot copy out of meta tensor),跳过搬运即可
  P5 pano_depth_utils.py utils3d 别名 + image_uv shim(正确实现是 GitHub utils3d_moge;
                        PyPI utils3d 无 numpy 子模块,新版又删了 UniK3D 时代 image_uv API)
  P6 general_utils.py   Vector3iVector 前显式 astype(np.int32):faces 是 int64(千万级三角形),
                        open3d int32 容器在 Windows 上对该转换直接 access violation(0xC0000005)
  P7 demo_scenegen.py   generate_world 前释放分层阶段模型(两个 FLUX 流水线 + SR + ZIM/GDINO,
                        几十 GB),否则大世界 mesh 组合阶段 np.vstack 内存不足(ArrayMemoryError)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------- upstream patches (idempotent)
def _patch_file(path: Path, old: str, new: str, name: str, log=print):
    if not path.is_file():
        raise SystemExit(f'[stage_a] 补丁目标不存在:{path}')
    s = path.read_text(encoding='utf-8')
    if new in s:
        log(f'[stage_a] 补丁 {name} 已在,跳过')
        return
    if s.count(old) != 1:
        raise SystemExit(f'[stage_a] 补丁 {name} 匹配失败({s.count(old)} 处 {old!r}):上游文件可能已改版,需人工核对 {path}')
    path.write_text(s.replace(old, new), encoding='utf-8')
    log(f'[stage_a] 补丁 {name} 已应用 → {path}')


_U3D_ALIAS = """try:
    import utils3d_moge as utils3d      # MoGe 官方依赖包名是 utils3d_moge,API 同 utils3d(microsoft/MoGe 内部同款别名)
except ImportError:
    import utils3d
if not hasattr(utils3d.numpy, 'image_uv'):
    # UniK3D 时代 API(新版已删):归一化 [0,1] 像素网格,(H, W, 2),喂 spherical_uv_to_directions
    import numpy as _np
    def _image_uv(width, height):
        u = _np.arange(width, dtype=_np.float32) + 0.5
        v = _np.arange(height, dtype=_np.float32) + 0.5
        uv = _np.stack(_np.meshgrid(u, v, indexing='xy'), axis=-1)
        return uv / _np.array([width, height], dtype=_np.float32)
    utils3d.numpy.image_uv = _image_uv
"""


_U3D_ALIAS_OLD = """try:
    import utils3d_moge as utils3d      # MoGe 官方依赖包名是 utils3d_moge,API 同 utils3d(microsoft/MoGe 内部同款别名)
except ImportError:
    import utils3d
"""


def _patch_utils3d_alias(path: Path, log=print):
    """P5:hy3dworld 直接 `import utils3d` 并调用 utils3d.numpy.intrinsics_to_fov / image_uv;
    正确实现来自 GitHub utils3d_moge(PyPI 的 utils3d 是另一个包,没有 numpy 子模块),
    新版 utils3d_moge 又删了 UniK3D 时代的 image_uv API——别名 + shim 一并注入。
    幂等标记 = shim 函数名 _image_uv(别名块可能已单独存在,须补 shim)。"""
    s = path.read_text(encoding='utf-8')
    if '_image_uv' in s:
        log('[stage_a] 补丁 P5 utils3d 别名+shim 已在,跳过')
        return
    bad = 'import utils3d\nimport utils3d.numpy  # 新版 utils3d 不自动加载子模块\n'
    if bad in s:                                   # 旧版 P5(错误方向)先归位
        s = s.replace(bad, 'import utils3d\n')
        log('[stage_a] 旧版 P5 已回退')
    if _U3D_ALIAS_OLD in s:                        # 有别名无 shim → 整段换新
        path.write_text(s.replace(_U3D_ALIAS_OLD, _U3D_ALIAS), encoding='utf-8')
        log(f'[stage_a] 补丁 P5 补 image_uv shim → {path}')
        return
    plain = 'import utils3d\n'
    if s.count(plain) != 1:
        raise SystemExit(f'[stage_a] 补丁 P5 匹配失败({s.count(plain)} 处 import utils3d):需人工核对 {path}')
    path.write_text(s.replace(plain, _U3D_ALIAS), encoding='utf-8')
    log(f'[stage_a] 补丁 P5 utils3d 别名+shim 已应用 → {path}')


def ensure_patches(repo: Path, log=print):
    _patch_file(repo / 'hy3dworld/utils/inpaint_utils.py',
                'pipe.enable_model_cpu_offload()', 'pipe.enable_sequential_cpu_offload()', 'P1 逐层顺序卸载', log)
    _patch_file(repo / 'hy3dworld/models/layer_decomposer.py',
                'sr_utils.build_sr_model(scale=self.scale, gpu_id=0)',
                'sr_utils.build_sr_model(scale=self.scale, gpu_id=0, tile=512)', 'P2 SR tile=512', log)
    _patch_file(repo / 'hy3dworld/utils/sr_utils.py',
                '    width = img.shape[1]',
                '    width = img.shape[1]\n    if width <= 1536:\n        upsampler.tile = 0  # 小图不分块(Windows tile_process 原生崩溃)',
                'P3 小图关 tile', log)
    _patch_file(repo / 'hy3dworld/utils/layer_utils.py',
                """        for model in models:
            if hasattr(model, 'cpu'):
                model.cpu()
            if hasattr(model, 'to'):
                model.to('cpu')""",
                """        for model in models:
            try:
                if hasattr(model, 'cpu'):
                    model.cpu()
                if hasattr(model, 'to'):
                    model.to('cpu')
            except (NotImplementedError, RuntimeError):
                pass            # 逐层顺序卸载下参数在 meta 设备,无需也无法搬运""",
                'P4 清理容错', log)
    _patch_utils3d_alias(repo / 'hy3dworld/utils/pano_depth_utils.py', log)
    _patch_file(repo / 'hy3dworld/utils/general_utils.py',
                'o3d.utility.Vector3iVector(faces.cpu().numpy())',
                'o3d.utility.Vector3iVector(faces.cpu().numpy().astype(np.int32))',
                'P6 faces int32', log)
    _patch_file(repo / 'demo_scenegen.py',
                '        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(',
                '        # 重建前释放分层阶段模型(两个 FLUX 流水线 + SR + ZIM/GDINO,几十 GB),\n'
                '        # 否则大世界 mesh 组合阶段 vstack 内存不足(ArrayMemoryError)\n'
                '        self.LayerDecomposer.inpaint_fg_model = None\n'
                '        self.LayerDecomposer.inpaint_sky_model = None\n'
                '        self.LayerDecomposer.sr_model = None\n'
                '        self.LayerDecomposer.zim_predictor = None\n'
                '        self.LayerDecomposer.gd_processor = None\n'
                '        self.LayerDecomposer.gd_model = None\n'
                '        import gc as _gc\n'
                '        _gc.collect()\n'
                '        separate_pano, fg_bboxes = self.hy3d_world._load_separate_pano_from_dir(',
                'P7 重建前释放模型', log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--repo', required=True, help='HunyuanWorld-1.0 仓库路径(demo_scenegen.py 所在)')
    ap.add_argument('--classes', default='outdoor', choices=('outdoor', 'indoor'))
    ap.add_argument('--fp8', action='store_true', help='开 FP8 权重量化(Windows 不可用:AngelSlim 原生崩溃;Linux 可试)')
    ap.add_argument('--fp8-attention', action='store_true', help='开 FP8 attention(需 SageAttention;Windows 默认关)')
    ap.add_argument('--labels-fg1', action='append', default=[], help='前景层 1 物体标签(可多次)')
    ap.add_argument('--labels-fg2', action='append', default=[], help='前景层 2 物体标签(可多次)')
    ap.add_argument('--no-patch', action='store_true', help='跳过 ensure_patches(上游已手工处理)')
    args = ap.parse_args()

    job = Path(args.job_dir)
    pano = job / 'pano.png'
    repo = Path(args.repo)
    if not pano.is_file():
        print(f'缺少全景 {pano}', file=sys.stderr)
        return 1
    demo = repo / 'demo_scenegen.py'
    if not demo.is_file():
        print(f'HunyuanWorld-1.0 仓库不对:{demo} 不存在', file=sys.stderr)
        return 1
    if not args.no_patch:
        ensure_patches(repo)

    out = job / 'scenegen'
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, '-u', str(demo), '--image_path', str(pano), '--classes', args.classes, '--output_path', str(out)]
    if args.fp8:
        cmd.append('--fp8_gemm')
    if args.fp8_attention:
        cmd.append('--fp8_attention')
    if args.labels_fg1:
        cmd += ['--labels_fg1', *args.labels_fg1]
    if args.labels_fg2:
        cmd += ['--labels_fg2', *args.labels_fg2]
    env = dict(os.environ)
    env['PYTHONUTF8'] = '1'              # Windows 控制台默认 GBK,上游打印 emoji 会 UnicodeEncodeError
    env['PYTHONPATH'] = str(repo) + os.pathsep + env.get('PYTHONPATH', '')
    print(f"[stage_a] 运行 {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo), env=env)
    if proc.returncode != 0:
        print(f'[stage_a] demo_scenegen 退出码 {proc.returncode}', file=sys.stderr)
        return proc.returncode

    layers = sorted(out.glob('mesh_layer*.ply'))
    if not layers:
        print(f'[stage_a] {out} 没有 mesh_layer*.ply 产物', file=sys.stderr)
        return 1
    rec = {'layers': [p.name for p in layers], 'classes': args.classes, 'fp8': bool(args.fp8),
           'labels_fg1': args.labels_fg1, 'labels_fg2': args.labels_fg2, 'seconds': round(time.time() - t0, 1),
           'coord_frame': 'z-up, pano center column = -x(hy3dworld spherical_uv_to_directions)'}
    (job / 'stage_a.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[stage_a] 完成:{len(layers)} 个分层 mesh,{rec["seconds"]}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

