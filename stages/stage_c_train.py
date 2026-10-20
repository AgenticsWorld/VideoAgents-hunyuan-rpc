#!/usr/bin/env python3
"""Stage C:多视图渲染 → 3DGS 训练(gsplat,MCMC 策略)→ 多档 SPZ。

输入 <job>/renders/(stage_b 产物:RGB + z 深度 + c2w/K,全部原始单位)。
训练在原始单位空间进行;标定不进训练,只在 world.json 体现(scale_fix)。
要点:
- 初始化点云 = 各视图深度反投影(比随机初始化稳,视图少时尤其),
  初始尺度取像素足迹 z / focal(免 KNN 的 O(N^2) 开销);
- MCMC 策略(cap_max 固定高斯数上限,显存可预期,3090 24G 按 1M 上限验证);
- 监督:RGB(0.8·L1 + 0.2·SSIM)+ 深度 L1(0.05,RGB+ED 的 ED 通道对输入 z 深度,无效像素掩掉);
- 结构与 gsplat examples/simple_trainer.py 对齐:ParameterDict + 每参数独立 Adam,
  means 学习率指数衰减,strategy.step_post_backward 返回值回写 state;
- 导出:训练参数直接按 SPZ 语义转换(scales 即 log_scale、opacities 即 logit、
  colors 经 sigmoid 得线性 RGB 再换算 SH DC 系数),quats 由 gsplat 的 wxyz 序转 SPZ 的 xyzw 序,
  经 gpu_worker/spz_writer.py 出 full_res,再随机子采样出 500k/150k/100k 档(同种子)。

产物:<job>/gs/gaussians.npz、<job>/gs/splats_<res>.spz、<job>/stage_c.json。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

def _patch_gsplat_flags(log=print):
    """Windows MSVC cl 不认 gsplat 自带的 -Wno-attributes(被 torch 转成 /Wno-attributes 后 D8021):
    幂等删掉该 flag(仅是告警抑制)。打在 env 的 site-packages 上,gsplat 重装后需重跑本阶段。"""
    try:
        import gsplat
    except ImportError:
        return
    import inspect
    backend = Path(inspect.getfile(gsplat)).parent / 'cuda' / '_backend.py'
    if not backend.is_file():
        return
    s = backend.read_text(encoding='utf-8')
    if '-Wno-attributes' not in s:
        return
    n = s.replace(', "-Wno-attributes"', '').replace('"-Wno-attributes", ', '').replace('"-Wno-attributes"', '')
    backend.write_text(n, encoding='utf-8')
    log(f'[stage_c] 补丁:gsplat 移除 -Wno-attributes → {backend}(torch_extensions 缓存需重建)')


SH_C0 = 0.28209479177387814
RES_TIERS = {'500k': 500_000, '150k': 150_000, '100k': 100_000}   # full_res = 全部


def load_dataset(renders: Path) -> dict:
    meta = json.loads((renders / 'cameras.json').read_text(encoding='utf-8'))
    from PIL import Image
    views = []
    for v in meta['views']:
        img = np.asarray(Image.open(renders / v['file']).convert('RGB'), dtype=np.float32) / 255.0
        depth = np.load(renders / v['depth']).astype(np.float32)
        views.append({'img': img, 'depth': depth, 'K': np.array(v['K'], dtype=np.float64).reshape(3, 3),
                      'c2w': np.array(v['c2w'], dtype=np.float64)})
    return {'views': views, 'width': meta['width'], 'height': meta['height'], 'gpo_raw': meta.get('gpo_raw')}


def init_points(data: dict, per_view: int = 15000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各视图深度反投影取点子样 → (N,3) 位置、(N,3) 颜色、(N,) 初始尺度(像素足迹 z/focal)。"""
    rng = np.random.default_rng(seed)
    pts, cols, footprint = [], [], []
    for v in data['views']:
        h, w = v['depth'].shape
        valid = np.isfinite(v['depth']) & (v['depth'] > 0)
        idx = np.flatnonzero(valid.ravel())
        if idx.size == 0:
            continue
        pick = rng.choice(idx, size=min(per_view, idx.size), replace=False)
        py, px = np.divmod(pick, w)
        z = v['depth'].ravel()[pick].astype(np.float64)
        fx, fy, cx, cy = v['K'][0, 0], v['K'][1, 1], v['K'][0, 2], v['K'][1, 2]
        x = (px - cx) / fx * z
        y = (py - cy) / fy * z
        cam = np.stack([x, y, z, np.ones_like(z)], axis=1)      # 相机系(y 下 z 前)
        world = (v['c2w'] @ cam.T).T[:, :3]
        pts.append(world)
        cols.append(v['img'][py, px])
        footprint.append(z / fx * 3.0)                        # 一个像素的世界尺寸 ×3:高斯须重叠成面,
                                                                # 否则亚像素级小高斯各管各的像素,新视角稀疏发糊
    return (np.concatenate(pts).astype(np.float32), np.concatenate(cols).astype(np.float32),
            np.concatenate(footprint).astype(np.float32))


def ssim_loss(a, b):
    """11x11 高斯窗 SSIM 损失(免 kornia/fused-ssim 依赖)。a,b: (1,3,H,W)。"""
    import torch
    import torch.nn.functional as F

    def gauss(win=11, sigma=1.5, device='cuda'):
        g = torch.exp(-((torch.arange(win, device=device) - win // 2) ** 2) / (2 * sigma ** 2))
        return (g / g.sum()).unsqueeze(0) * (g / g.sum()).unsqueeze(1)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = gauss(device=a.device).expand(3, 1, 11, 11)
    mu_a, mu_b = (F.conv2d(x, w, padding=11 // 2, groups=3) for x in (a, b))
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, w, padding=11 // 2, groups=3) - mu_a2
    sb = F.conv2d(b * b, w, padding=11 // 2, groups=3) - mu_b2
    sab = F.conv2d(a * b, w, padding=11 // 2, groups=3) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return 1.0 - ssim_map.mean()


def train(data: dict, *, steps: int, max_gaussians: int, log=print) -> dict:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    device = 'cuda'
    pts, cols, footprint = init_points(data)
    log(f'[stage_c] 初始化 {pts.shape[0]} 点(深度反投影)')
    splats = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(torch.from_numpy(pts).to(device)),
        'quats': torch.nn.Parameter(torch.eye(4)[0].repeat(pts.shape[0], 1).to(device)),   # (1,0,0,0) wxyz
        'scales': torch.nn.Parameter(torch.log(torch.from_numpy(footprint).clamp_min(1e-6))
                                     .unsqueeze(1).repeat(1, 3).to(device)),
        'opacities': torch.nn.Parameter(torch.logit(torch.full((pts.shape[0],), 0.5, device=device))),
        'colors': torch.nn.Parameter(torch.logit(torch.from_numpy(cols).clamp(1e-4, 1 - 1e-4).to(device))),
    }).to(device)
    base_lr = {'means': 1.6e-4, 'scales': 5e-3, 'quats': 1e-3, 'opacities': 5e-2, 'colors': 2.5e-3}
    optimizers = {name: torch.optim.Adam([p], lr=base_lr[name], eps=1e-15) for name, p in splats.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1.0 / max(steps, 1)))
    strategy = MCMCStrategy(verbose=False, cap_max=max_gaussians)
    strategy.check_sanity(splats, optimizers)
    scene_scale = float(np.linalg.norm(pts.std(axis=0)) * 3)
    import inspect                                                  # gsplat 版本差异:新版 MCMCStrategy.initialize_state 不收 scene_scale
    if 'scene_scale' in inspect.signature(strategy.initialize_state).parameters:
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy_state = strategy.initialize_state()

    H, W = data['height'], data['width']
    Ks = torch.tensor(np.stack([v['K'] for v in data['views']]), dtype=torch.float32, device=device)
    w2c = torch.tensor(np.stack([np.linalg.inv(v['c2w']) for v in data['views']]), dtype=torch.float32, device=device)
    imgs = [torch.from_numpy(v['img']).permute(2, 0, 1).unsqueeze(0).to(device) for v in data['views']]
    depths = [torch.from_numpy(np.nan_to_num(v['depth'], nan=0.0, posinf=0.0)).to(device) for v in data['views']]
    depth_masks = [d > 0 for d in depths]
    sched = max(1, steps // 5)

    t0 = time.time()
    for step in range(steps):
        i = int(torch.randint(0, len(imgs), (1,)).item())
        render, _alphas, info = rasterization(means=splats['means'], quats=splats['quats'],
                                              scales=torch.exp(splats['scales']), opacities=torch.sigmoid(splats['opacities']),
                                              colors=torch.sigmoid(splats['colors']),
                                              viewmats=w2c[i:i + 1], Ks=Ks[i:i + 1], width=W, height=H,
                                              render_mode='RGB+ED', packed=False)
        rgb = render[..., :3].permute(0, 3, 1, 2)
        dep = render[..., 3]
        l1 = (rgb - imgs[i]).abs().mean()
        ss = ssim_loss(rgb, imgs[i])
        dm = depth_masks[i]
        dloss = (dep[0][dm] - depths[i][dm]).abs().mean() if dm.any() else torch.tensor(0.0, device=device)
        # 尺度正则(2026-10-20,对齐 HY-World 2.0 world_gs_trainer 的 use_scale_regularization):
        # 抑制拉长针状高斯(深度突变边缘的杂刺),惩罚对数尺度极差超过阈值的部分
        ls = splats['scales']                                   # log 尺度
        ratio = ls.max(dim=1).values - ls.min(dim=1).values     # log(最长轴/最短轴)
        scale_reg = torch.relu(ratio - math.log(5.0)).mean()
        loss = 0.8 * l1 + 0.2 * ss + 0.05 * dloss + 0.01 * scale_reg
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy_kw = dict(params=splats, optimizers=optimizers, state=strategy_state,
                           step=step, info=info, packed=False, lr=base_lr['means'])
        sig_params = inspect.signature(strategy.step_post_backward).parameters
        ret = strategy.step_post_backward(**{k: v for k, v in strategy_kw.items() if k in sig_params})
        if ret is not None:                             # 有的版本返回 state,有的原地修改返回 None
            strategy_state = ret
        if step % sched == 0 or step == steps - 1:
            log(f'[stage_c] step {step + 1}/{steps} loss={loss.item():.4f}(l1={l1.item():.4f} ssim={ss.item():.4f} '
                f'depth={dloss.item():.4f}) N={splats["means"].shape[0]} {time.time() - t0:.0f}s')
    return {'n_final': int(splats['means'].shape[0]), 'seconds': round(time.time() - t0, 1),
            'params': {name: p.detach().cpu().numpy() for name, p in splats.items()}}


def export_spz(params: dict, out: Path, *, seed: int = 0, log=print) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spz_writer import write_spz

    means = params['means'].astype(np.float32)
    quats_xyzw = params['quats'].astype(np.float32)[:, [1, 2, 3, 0]]   # gsplat wxyz → SPZ xyzw
    log_scales = params['scales'].astype(np.float32)                   # render 用 exp(scales) ⇒ 即 SPZ log_scale
    logits = params['opacities'].astype(np.float32)                    # render 用 sigmoid(opacities) ⇒ 即 SPZ logit
    rgb = 1.0 / (1.0 + np.exp(-params['colors'].astype(np.float32)))
    dc = (rgb - 0.5) / SH_C0                                           # SPZ colors = SH DC 系数
    files: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    n = means.shape[0]
    for res, cap in [('full_res', None), *RES_TIERS.items()]:
        idx = np.arange(n) if cap is None or n <= cap else rng.choice(n, size=cap, replace=False)
        name = f'splats_{res}.spz'
        write_spz(out / name, means[idx], quats_xyzw[idx], log_scales[idx], logits[idx], dc[idx],
                  antialiased=True)
        files[res] = name
        log(f'[stage_c] {name}:{len(idx)} splats,{(out / name).stat().st_size // 1024 // 1024} MB')
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', required=True)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--max-gaussians', type=int, default=1_000_000, help='MCMC cap_max')
    ap.add_argument('--smoke', action='store_true', help='袖珍自检(200 步)')
    args = ap.parse_args()
    if args.smoke:
        args.steps = 200

    job = Path(args.job_dir)
    out = job / 'gs'
    out.mkdir(exist_ok=True)
    _patch_gsplat_flags()
    data = load_dataset(job / 'renders')
    result = train(data, steps=args.steps, max_gaussians=args.max_gaussians)
    np.savez(out / 'gaussians.npz', **result['params'])
    files = export_spz(result['params'], out)
    rec = {'steps': args.steps, 'n_init': int(result['params']['means'].shape[0]), 'n_final': result['n_final'],
           'seconds': result['seconds'], 'splats': files, 'gpo_raw': data.get('gpo_raw')}
    (job / 'stage_c.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[stage_c] 完成:{rec["n_final"]} 高斯,{rec["seconds"]}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
