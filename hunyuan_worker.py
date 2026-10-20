#!/usr/bin/env python3
"""HunyuanWorld GPU worker(2026-10-10):跑在 GPU 主机(默认 Windows + 3090)上的小型 HTTP 服务,
把「场景全景 → HunyuanWorld-1.0 分层重建 → 多视图渲染 → 3DGS 训练 → SPZ/GLB 产物」四阶段流水线
包装成任务 API,供 VideoAgents 主仓的 modules/hunyuan_world.py 远程调用。部署见 docs/hunyuan_deploy.md。

设计要点:
- 自包含:只依赖 fastapi/uvicorn 与 hyworld conda env,不 import VideoAgents 其他模块;
  各阶段是 stages/ 下的独立子进程脚本(用 config.json 里的 env python 启动),日志实时回传。
- 单卡单任务:同一时刻只跑一个任务,忙时 POST /worlds 回 409。
- 断点续跑:每阶段写完 <job>/stages/<n>_<name>.done 标记;POST /worlds/<id>/resume 从首个缺失标记的阶段重跑。
- TTL 清理:完成/失败/取消超过 job_ttl_days 天的任务目录后台自动删。

端点:
  GET    /health                      GPU/env/磁盘/权重自检(models.html「验证」按钮、CLI --check 用)
  POST   /worlds                      multipart:pano 文件 + params JSON → {job_id};忙 409
  GET    /worlds/{job_id}             {status, stage, log_tail, error, artifacts}
  GET    /worlds/{job_id}/artifacts   产物清单 [{name, size}]
  GET    /worlds/{job_id}/artifacts/{name}  下载单个产物
  POST   /worlds/{job_id}/resume      断点续跑
  DELETE /worlds/{job_id}             取消(运行中)或立即清理(已结束)
鉴权:config.json#token 非空时,请求须带 X-Worker-Token 头。只绑内网。

启动:python hunyuan_worker.py [--config config.json] [--port 8310]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse
except ImportError as e:  # noqa: BLE001
    print(f'缺少依赖:{e}\n先 pip install -r requirements.txt(见 gpu_worker/README.md)', file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent
VERSION = 'hunyuan-worker 0.2.0'
STAGES = ('scenegen', 'render', 'gs', 'pack')
STAGE_SCRIPT = {'scenegen': 'stage_a_scenegen.py', 'render': 'stage_b_render.py', 'gs': 'stage_c_train.py', 'pack': 'stage_d_pack.py'}
LOG_TAIL = 80


# ---------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    cfg.setdefault('hyworld_repo', '')           # HunyuanWorld-1.0 仓库路径(demo_scenegen.py 所在)
    cfg.setdefault('python', sys.executable)     # hyworld conda env 的 python
    cfg.setdefault('work_root', str(HERE / 'jobs'))
    cfg.setdefault('port', 8310)
    cfg.setdefault('token', '')
    cfg.setdefault('job_ttl_days', 7)
    cfg.setdefault('defaults', {})               # fp8 / views / gs_steps / classes 等缺省任务参数
    cfg['hyworld_repo'] = os.path.expandvars(os.path.expanduser(cfg['hyworld_repo']))
    cfg['python'] = os.path.expandvars(os.path.expanduser(cfg['python']))
    cfg['work_root'] = os.path.expandvars(os.path.expanduser(cfg['work_root']))
    return cfg


CFG: dict = {}
JOBS: dict[str, dict] = {}                       # job_id -> 内存态(与 job.json 同步)
LOCK = threading.Lock()                          # JOBS 写锁
RUNNING: dict[str, threading.Event] = {}         # job_id -> cancel event(仅运行中)
QUEUE: queue.Queue[str] = queue.Queue()          # 待跑 job_id(容量实际 ≤1:忙时直接 409)


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def job_dir(job_id: str) -> Path:
    return Path(CFG['work_root']) / 'jobs' / job_id


def _save(job: dict):
    job['updated_at'] = _now()
    d = job_dir(job['job_id'])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / 'job.json.tmp'
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, d / 'job.json')


def _load(job_id: str) -> dict | None:
    p = job_dir(job_id) / 'job.json'
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            return None
    return None


def get_job(job_id: str) -> dict:
    with LOCK:
        job = JOBS.get(job_id) or _load(job_id)
        if job:
            JOBS[job_id] = job
    if not job:
        raise HTTPException(404, f'没有任务 {job_id}')
    return job


def jlog(job: dict, line: str):
    line = line.rstrip()
    if not line:
        return
    with LOCK:
        job.setdefault('log_tail', []).append(line)
        job['log_tail'] = job['log_tail'][-LOG_TAIL:]
    with open(job_dir(job['job_id']) / 'job.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------- stage runner
def _stage_env() -> dict:
    env = dict(os.environ)
    # 显存碎片缓解(2026-10-19):顺序卸载/分割模型常驻下,VAE 编码等大分配易被碎片卡死(报大量空闲却分配失败)
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('PYTHONUTF8', '1')                 # Windows 控制台默认 GBK,上游 emoji 输出会 UnicodeEncodeError
    if CFG.get('hf_home'):
        env['HF_HOME'] = CFG['hf_home']             # HF 缓存重定向(默认在 C 盘用户目录,FLUX 几十 GB 容易写满)
    if CFG.get('hf_endpoint'):
        env['HF_ENDPOINT'] = CFG['hf_endpoint']
    if CFG.get('hf_token'):
        env['HF_TOKEN'] = CFG['hf_token']
    return env


_PROGRESS_RE = re.compile(r'\d+%\||\d+\.\d+\s*[MGK]?B/s|it/s\]|s/it\]')


def _pump_output(job: dict, proc: subprocess.Popen, cancel: threading.Event):
    """按 \\r 与 \\n 双流切子进程输出:普通行进 job.log;tqdm 进度条(\\r 刷新不产生换行)
    写 job['progress'] 并按 5s 节流提交日志——大文件下载期间客户端/页面也能看到进度(2026-10-19)。"""
    buf = ''
    last_commit = 0.0
    assert proc.stdout is not None
    while True:
        if cancel.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            jlog(job, '[worker] 任务取消,stage 已终止')
            return
        try:
            chunk = proc.stdout.read(4096)
        except Exception:  # noqa: BLE001
            break
        if chunk:
            buf += chunk
            while True:
                m = re.search(r'[\r\n]', buf)
                if not m:
                    break
                seg, buf = buf[:m.start()], buf[m.end():]
                seg = seg.strip()
                if not seg:
                    continue
                if _PROGRESS_RE.search(seg) and len(seg) < 400:
                    job['progress'] = seg[-200:]
                    if time.time() - last_commit > 5:
                        jlog(job, f'⏳ {seg[-160:]}')
                        last_commit = time.time()
                else:
                    jlog(job, seg)
            continue
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    if buf.strip():
        jlog(job, buf.strip())
    job['progress'] = None


def _run_stage(job: dict, stage: str, cancel: threading.Event) -> bool:
    """跑一个阶段子进程;返回是否成功。日志逐行进 job.log / log_tail,进度条进 job['progress']。"""
    d = job_dir(job['job_id'])
    script = HERE / 'stages' / STAGE_SCRIPT[stage]
    params = job['params']
    cmd = [CFG['python'], '-u', str(script), '--job-dir', str(d)]
    if stage == 'scenegen':
        cmd += ['--repo', CFG['hyworld_repo'], '--classes', params.get('classes') or 'outdoor']
        if params.get('fp8', True):
            cmd.append('--fp8')
        if params.get('fp8_attention', False):
            cmd.append('--fp8-attention')
        for label in params.get('labels_fg1') or []:
            cmd += ['--labels-fg1', label]
        for label in params.get('labels_fg2') or []:
            cmd += ['--labels-fg2', label]
    elif stage == 'render':
        cmd += ['--views', str(int(params.get('views') or 24)),
                '--camera-height-m', str(float(params.get('camera_height_m') or 1.6)),
                '--radius-m', str(float(params.get('radius_m') or 0.35)),
                '--width', str(int(params.get('render_width') or 1600)),
                '--height', str(int(params.get('render_height') or 900))]
    elif stage == 'gs':
        cmd += ['--steps', str(int(params.get('gs_steps') or 8000)),
                '--max-gaussians', str(int(params.get('max_gaussians') or 1000000))]
        if params.get('smoke'):
            cmd.append('--smoke')
    job['stage'] = stage
    _save(job)
    jlog(job, f'===== stage {stage} =====')
    proc = subprocess.Popen(cmd, cwd=str(d), env=_stage_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)
    _pump_output(job, proc, cancel)
    if cancel.is_set():
        return False
    rc = proc.returncode
    if rc != 0:
        jlog(job, f'[worker] stage {stage} 失败,退出码 {rc}')
        return False
    (d / 'stages').mkdir(exist_ok=True)
    (d / 'stages' / f'{STAGES.index(stage) + 1}_{stage}.done').write_text(_now(), encoding='utf-8')
    return True


def _collect_artifacts(job: dict):
    d = job_dir(job['job_id'])
    manifest = d / 'artifacts.json'
    job['artifacts'] = {}
    job['pack_meta'] = {}
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            job['artifacts'] = data.get('artifacts') or {}
            job['pack_meta'] = data.get('meta') or {}
        except Exception:  # noqa: BLE001
            pass


def _run_job(job_id: str, cancel: threading.Event, start_at: int = 0):
    job = get_job(job_id)
    try:
        for i, stage in enumerate(STAGES):
            if i < start_at or (job_dir(job_id) / 'stages' / f'{i + 1}_{stage}.done').is_file():
                if i >= start_at:
                    jlog(job, f'[worker] stage {stage} 已有完成标记,跳过')
                continue
            if not _run_stage(job, stage, cancel):
                job['status'] = 'cancelled' if cancel.is_set() else 'error'
                job['error'] = '已取消' if cancel.is_set() else f'stage {stage} 失败(见日志)'
                job['finished_at'] = _now()
                _save(job)
                return
        _collect_artifacts(job)
        job['status'] = 'done'
        job['stage'] = None
        job['finished_at'] = _now()
        _save(job)
        jlog(job, '[worker] 全部阶段完成')
    except Exception as e:  # noqa: BLE001
        job['status'] = 'error'
        job['error'] = str(e)[:500]
        job['finished_at'] = _now()
        _save(job)
        jlog(job, f'[worker] 异常:{e}')
    finally:
        RUNNING.pop(job_id, None)


def _runner_loop():
    while True:
        job_id = QUEUE.get()
        cancel = RUNNING.get(job_id)
        if cancel is None:
            continue
        _run_job(job_id, cancel)


def _ttl_sweeper():
    while True:
        time.sleep(3600)
        try:
            ttl_s = float(CFG.get('job_ttl_days') or 7) * 86400
            root = job_dir('').parent  # work_root/jobs
            if not root.is_dir():
                continue
            now = time.time()
            for d in root.iterdir():
                job = _load(d.name)
                if not job or job.get('status') in ('queued', 'running'):
                    continue
                fin = job.get('finished_at') or job.get('updated_at') or ''
                try:
                    age = now - time.mktime(time.strptime(fin, '%Y-%m-%dT%H:%M:%S'))
                except Exception:  # noqa: BLE001
                    continue
                if age > ttl_s:
                    shutil.rmtree(d, ignore_errors=True)
                    with LOCK:
                        JOBS.pop(d.name, None)
        except Exception as e:  # noqa: BLE001
            print(f'[worker] TTL 清理异常:{e}', flush=True)


# ---------------------------------------------------------------- health
_HEALTH_CACHE = {'at': 0.0, 'data': None}


def health() -> dict:
    if time.time() - _HEALTH_CACHE['at'] < 60 and _HEALTH_CACHE['data']:
        return _HEALTH_CACHE['data']
    checks: dict = {'version': VERSION, 'busy': bool(RUNNING)}
    checks['python'] = {'path': CFG['python'], 'ok': Path(CFG['python']).is_file()}
    repo = Path(CFG['hyworld_repo']) if CFG['hyworld_repo'] else None
    checks['hyworld_repo'] = {'path': CFG['hyworld_repo'], 'ok': bool(repo and (repo / 'demo_scenegen.py').is_file())}
    if not checks['hyworld_repo']['ok']:
        checks['hyworld_repo']['hint'] = '该目录下需存在 demo_scenegen.py:请先 clone HunyuanWorld-1.0,或把 hyworld_repo 改成实际路径(注意多套一层目录的情况)'
    try:
        usage = shutil.disk_usage(CFG['work_root'] if Path(CFG['work_root']).exists() else '.')
        checks['disk_free_gb'] = round(usage.free / 1e9, 1)
    except Exception:  # noqa: BLE001
        checks['disk_free_gb'] = None
    try:
        probe = subprocess.run([CFG['python'], '-c',
                                'import torch,json;print(json.dumps({"cuda":torch.cuda.is_available(),'
                                '"name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",'
                                '"vram_gb":round(torch.cuda.get_device_properties(0).total_memory/1e9,1) if torch.cuda.is_available() else 0}))'],
                               capture_output=True, text=True, timeout=60)
        checks['gpu'] = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else {
            'cuda': False, 'error': (probe.stderr or probe.stdout)[-300:]}
    except Exception as e:  # noqa: BLE001
        checks['gpu'] = {'cuda': False, 'error': str(e)[:300]}
    # 全链路依赖自检(2026-10-19):逐模块 import,哪个缺/版本不兼容一眼可见;失败不拖垮 health
    probe_src = ('import importlib,json\n'
                 'mods = ["torch","diffusers","transformers","utils3d","moge","zim_anything","basicsr","realesrgan",'
                 '"cv2","PIL","numpy","open3d","trimesh","gsplat","onnxruntime","skimage"]\n'
                 'out = {}\n'
                 'for m in mods:\n'
                 '    try:\n'
                 '        importlib.import_module(m)\n'
                 '        out[m] = True\n'
                 '    except Exception as e:\n'
                 '        out[m] = f"{type(e).__name__}: {str(e)[:160]}"\n'
                 'print("IMPORT_PROBE" + json.dumps(out))\n')
    try:
        probe = subprocess.run([CFG['python'], '-c', probe_src], capture_output=True, text=True, timeout=120)
        line = next((l for l in probe.stdout.splitlines() if l.startswith('IMPORT_PROBE')), None)
        checks['imports'] = json.loads(line[len('IMPORT_PROBE'):]) if line else {
            'error': (probe.stderr or probe.stdout)[-400:]}
    except Exception as e:  # noqa: BLE001
        checks['imports'] = {'error': str(e)[:300]}
    bad_imports = {m: v for m, v in (checks.get('imports') or {}).items() if v is not True}
    checks['ok'] = bool(checks['python']['ok'] and checks['hyworld_repo']['ok'] and checks['gpu'].get('cuda')
                        and not bad_imports)
    if bad_imports:
        checks['hint'] = '依赖自检未过:' + '; '.join(f'{m} → {v}' for m, v in bad_imports.items())[:600]
    _HEALTH_CACHE.update({'at': time.time(), 'data': checks})
    return checks


# ---------------------------------------------------------------- http
app = FastAPI(title='HunyuanWorld GPU worker', version=VERSION)


@app.middleware('http')
async def auth(request: Request, call_next):
    token = CFG.get('token') or ''
    if token and request.headers.get('x-worker-token', '') != token:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': 'X-Worker-Token 不对或缺失'}, status_code=401)
    return await call_next(request)


@app.get('/health')
def get_health():
    return health()


@app.post('/worlds')
async def submit(pano: UploadFile = File(...), params: str = Form('{}')):
    if RUNNING or not QUEUE.empty():
        raise HTTPException(409, 'GPU 主机忙:有任务正在运行,稍后再试')
    if not CFG['hyworld_repo']:
        raise HTTPException(500, 'worker 未配置 hyworld_repo')
    try:
        p = json.loads(params or '{}')
    except json.JSONDecodeError:
        raise HTTPException(400, 'params 须为 JSON 字符串') from None
    merged = {**(CFG.get('defaults') or {}), **p}
    job_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    content = await pano.read()
    if len(content) < 1024:
        raise HTTPException(400, 'pano 文件太小,不像全景图')
    (d / 'pano.png').write_bytes(content)
    job = {'job_id': job_id, 'status': 'queued', 'stage': None, 'params': merged,
           'created_at': _now(), 'updated_at': _now(), 'finished_at': None, 'error': None,
           'log_tail': [f'任务 {job_id} 已创建,全景 {len(content) // 1024} KB,参数 {json.dumps(merged, ensure_ascii=False)}'],
           'artifacts': {}}
    with LOCK:
        JOBS[job_id] = job
    _save(job)
    RUNNING[job_id] = threading.Event()
    QUEUE.put(job_id)
    return {'job_id': job_id}


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ('job_id', 'status', 'stage', 'progress', 'params', 'created_at', 'finished_at', 'error',
                                    'log_tail', 'artifacts', 'pack_meta')}


@app.get('/worlds/{job_id}')
def status(job_id: str):
    return _public(get_job(job_id))


@app.get('/worlds/{job_id}/artifacts')
def artifacts(job_id: str):
    job = get_job(job_id)
    d = job_dir(job_id)
    names: list[str] = []
    for v in (job.get('artifacts') or {}).values():
        names.extend(v.values() if isinstance(v, dict) else [v])
    return {'artifacts': [{'name': n, 'size': (d / n).stat().st_size} for n in names if (d / n).is_file()]}


@app.get('/worlds/{job_id}/artifacts/{name:path}')
def artifact(job_id: str, name: str):
    get_job(job_id)
    d = job_dir(job_id)
    target = (d / name).resolve()
    if not str(target).startswith(str(d.resolve())) or not target.is_file():
        raise HTTPException(404, f'没有产物 {name}')
    return FileResponse(str(target), filename=Path(name).name)


@app.post('/worlds/{job_id}/resume')
def resume(job_id: str, from_stage: str | None = None):
    job = get_job(job_id)
    if job['status'] in ('queued', 'running'):
        raise HTTPException(409, '任务正在运行')
    if RUNNING:
        raise HTTPException(409, 'GPU 主机忙:有任务正在运行')
    d = job_dir(job_id)
    if from_stage:
        if from_stage not in STAGES:
            raise HTTPException(400, f'from_stage 须为 {STAGES} 之一')
        # 强制重跑:删掉该阶段及之后的完成标记(2026-10-20,调试用)
        for i in range(STAGES.index(from_stage), len(STAGES)):
            marker = d / 'stages' / f'{i + 1}_{STAGES[i]}.done'
            marker.unlink(missing_ok=True)
    start_at = 0
    for i, stage in enumerate(STAGES):
        if (d / 'stages' / f'{i + 1}_{stage}.done').is_file():
            start_at = i + 1
    if start_at >= len(STAGES):
        _collect_artifacts(job)
        job['status'] = 'done'
        _save(job)
        return {'ok': True, 'resumed': False, 'status': 'done'}
    job['status'] = 'queued'
    job['error'] = None
    _save(job)
    jlog(job, f'[worker] 从 stage {STAGES[start_at]} 续跑')
    cancel = threading.Event()
    RUNNING[job_id] = cancel
    threading.Thread(target=_run_job, args=(job_id, cancel, start_at), daemon=True).start()
    return {'ok': True, 'resumed': True, 'from_stage': STAGES[start_at]}


@app.delete('/worlds/{job_id}')
def delete(job_id: str, purge: bool = False):
    job = get_job(job_id)
    ev = RUNNING.get(job_id)
    if ev is not None:
        ev.set()
        return {'ok': True, 'cancelled': True}
    if purge:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
        with LOCK:
            JOBS.pop(job_id, None)
        return {'ok': True, 'purged': True}
    return {'ok': True, 'cancelled': False, 'status': job['status']}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(HERE / 'config.json'))
    ap.add_argument('--port', type=int, default=None)
    args = ap.parse_args()
    global CFG
    CFG = load_config(Path(args.config))
    Path(CFG['work_root']).mkdir(parents=True, exist_ok=True)
    jobs_root = Path(CFG['work_root']) / 'jobs'
    jobs_root.mkdir(exist_ok=True)
    # 崩溃恢复(2026-10-19):worker 重启后,卡在 queued/running 的旧任务标记为 error 可续跑
    # (阶段 .done 标记仍在,POST /worlds/<id>/resume 从断点继续;子进程已随 worker 一起消失)
    for d in jobs_root.iterdir():
        job = _load(d.name)
        if job and job.get('status') in ('queued', 'running'):
            job['status'] = 'error'
            job['error'] = 'worker 重启,任务中断(可 resume 续跑)'
            job['finished_at'] = _now()
            _save(job)
            jlog(job, '[worker] 检测到中断任务,已标记为可续跑')
    threading.Thread(target=_runner_loop, daemon=True).start()
    threading.Thread(target=_ttl_sweeper, daemon=True).start()
    port = int(args.port or CFG['port'])
    print(f'[worker] {VERSION} 监听 0.0.0.0:{port},work_root={CFG["work_root"]},repo={CFG["hyworld_repo"]}', flush=True)
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='warning')
    return 0


if __name__ == '__main__':
    sys.exit(main())
