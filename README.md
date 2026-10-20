# hunyuan-rpc — HunyuanWorld GPU worker

跑在你自己的 GPU 主机(默认 **Windows + RTX 3090 24G**)上的小型 HTTP 服务,把
「场景全景 → [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) 分层 3D 重建 →
多视图渲染 → 3DGS 训练 → SPZ/GLB 产物」包装成任务 API。

本服务是 [VideoAgents](https://github.com/AgenticsWorld/VideoAgents)「🌍 世界模型 ·
HunyuanWorld(本地算力)」渠道的远程执行端;VideoAgents 主仓只含 HTTP 客户端
(`modules/hunyuan_world.py`),功能与产物契约见主仓
[docs/hunyuan_world.md](https://github.com/AgenticsWorld/VideoAgents/blob/dev/docs/hunyuan_world.md)。
本仓库自包含,不 import 主仓任何模块。

## 流水线

| 阶段 | 内容 |
|---|---|
| A 重建 | `demo_scenegen.py`(bf16 + 逐层顺序卸载)对场景全景出分层 textured mesh;上游仓库的兼容补丁由 `stages/stage_a_scenegen.py` 每次运行前幂等自动应用(见下文「必打补丁」) |
| B 渲染 | open3d `RaycastingScene` CPU 光线投射(绕开 Filament/GL 在部分 Windows 主机上的离屏渲染故障),有界轨迹多视图 RGB+深度 |
| C 训练 | gsplat(MCMC 策略)+ 尺度正则 → `splats_{full_res,500k,150k,100k}.spz`(内置纯 Python SPZ v2 编码器 `spz_writer.py`,免编译) |
| D 打包 | `collider.glb` 减面碰撞体、`thumbnail.jpg`、`world_pano.jpg`、产物清单 |

## 端点

| 端点 | 语义 |
|---|---|
| `GET /health` | GPU/env/磁盘/全链路 import 自检 + 版本 |
| `POST /worlds` | multipart:pano + params JSON → `{job_id}`;单卡单任务,忙 409 |
| `GET /worlds/{id}` | `{status, stage, progress, log_tail, error, artifacts}`(progress = 下载/训练进度条) |
| `GET /worlds/{id}/artifacts/{name:path}` | 下载产物(支持子路径) |
| `POST /worlds/{id}/resume[?from_stage=scenegen\|render\|gs\|pack]` | 断点续跑 / 强跑指定阶段 |
| `DELETE /worlds/{id}` | 取消或清理 |

鉴权:`config.json#token` 非空时须带 `X-Worker-Token` 头。**只绑内网。**

## 部署(默认 Windows 10/11 + 3090 24G)

### 0. 前置

| 项 | 要求 |
|---|---|
| GPU | NVIDIA,≥24 GB 显存(3090/4090 验证口径;更小显存需下调 `max_gaussians` 与视图分辨率) |
| 内存 | 64 GB,且**虚拟内存须加大到 ≥40 GB 并重启生效**(见故障表「无声 -1」) |
| 磁盘 | ≥100 GB 空闲(权重 ~60 GB + 任务工作区) |
| 软件 | Git、Miniconda、VS Build Tools(C++ 桌面开发,gsplat JIT 编译用)、CUDA Toolkit 12.4 |
| 网络 | 能访问 huggingface.co(不能则配 `HF_ENDPOINT` 镜像) |

### 1. 拿代码

```bat
git clone https://github.com/AgenticsWorld/VideoAgents-hunyuan-rpc.git D:\hunyuan-rpc
```

### 2. 建环境(Anaconda Prompt)

```bat
conda create -n hyworld1 python=3.10 -y
conda activate hyworld1

:: PyTorch 2.5 + cu124(与 HunyuanWorld-1.0 官方测试版本一致)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

:: HunyuanWorld-1.0 依赖(版本与官方 docker/HunyuanWorld.yaml 对齐;
:: transformers 不要装新版——新版对 torch 2.5 禁用 PyTorch 支持,
:: 会报 "Failed to import diffusers.loaders.unet ... name 'nn' is not defined")
pip install diffusers==0.34.0 transformers==4.51.0 accelerate==1.6.0 peft==0.15.0 safetensors==0.5.3
pip install open-clip-torch==2.30.0 pytorch-lightning==2.4.0 torchmetrics==1.7.1
pip install opencv-python imageio imageio-ffmpeg tqdm sentencepiece
pip install scikit-image matplotlib easydict
pip install "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git"
:: 注意:MoGe 深度用的是 GitHub 的 utils3d_moge(含 numpy 子模块),不是 PyPI 的 utils3d 包

pip install basicsr-fixed facexlib gfpgan realesrgan
:: basicsr/realesrgan 与 torchvision 0.18+ 的兼容 shim(否则 import 报 functional_tensor 缺失)
python -c "import pathlib, torchvision; p = pathlib.Path(torchvision.__file__).parent/'transforms'/'functional_tensor.py'; p.write_text('from ._functional_tensor import *\n', encoding='utf-8')"
pip install onnxruntime-gpu                               :: ZIM 分割(ViT-L onnx)
pip install open3d>=0.18 trimesh                          :: Stage B 渲染 / Stage D 打包(0.17 有原生崩溃,实测 0.20 修复)
pip install moge                                          :: 全景深度

:: 3DGS 训练:无 torch 2.5+Windows 预编译 wheel,须 JIT 编译一次:
:: 同一 Anaconda Prompt 里 call <VS>\VC\Auxiliary\Build\vcvars64.bat(vswhere 可查路径),
:: where cl 确认后跑一次触发命令(见故障表),产物入缓存后普通环境直接用
pip install gsplat

:: worker 服务自身
pip install -r D:\hunyuan-rpc\requirements.txt
```

### 3. 拉 HunyuanWorld-1.0 仓库与权重

```bat
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git D:\models\HunyuanWorld-1.0

:: ZIM 分割:仓库必须克隆在 HunyuanWorld-1.0 目录内(代码用相对路径 ./ZIM/zim_vit_l_2092 找权重)
cd /d D:\models\HunyuanWorld-1.0
git clone https://github.com/naver-ai/ZIM.git ZIM
cd ZIM
pip install -e .
mkdir zim_vit_l_2092
cd zim_vit_l_2092
curl -L -O https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/encoder.onnx
curl -L -O https://huggingface.co/naver-iv/zim-anything-vitl/resolve/main/zim_vit_l_2092/decoder.onnx
```

**FLUX 是 gated 仓库(必做)**:先用 HF 账号在浏览器里接受
[FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) 与
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 的许可,再
`huggingface-cli login --token <HF_TOKEN>`(read 权限即可),否则首次运行报
`GatedRepoError: 401/403`。也可把 token 写进 `config.json#hf_token`(worker 注入阶段子进程)。

预下载权重(带进度条,`set HF_HOME` 先行):

```bat
set HF_HOME=D:\hf_cache
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev
huggingface-cli download tencent/HunyuanWorld-1
huggingface-cli download Ruicheng/moge-vitl
huggingface-cli download IDEA-Research/grounding-dino-tiny
```

### 4. 必打补丁(worker 自动管理)

`stages/stage_a_scenegen.py` 每次运行前**幂等自动应用**,无需手工;此处仅作原理记录
(全部来自 Windows + 3090 实机踩坑):

- **P1 显存策略**:`enable_model_cpu_offload` → `enable_sequential_cpu_offload`(`inpaint_utils.py`)。
  整模型卸载会把 24GB bf16 transformer 整体搬上 GPU;逐层卸载峰值仅几 GB,bf16 直跑,
  绕开 AngelSlim FP8(后者在 Windows torch 2.5 必现 access violation,位置随机漂移,勿用)。
- **P2 超分分块**:`build_sr_model` 加 `tile=512`(`layer_decomposer.py`)。RealESRGAN 大图不分块必 OOM,
  且 OOM 被 `sr_inference` 吞掉只打印不存盘,到天空阶段才报 `full_image_sr.png` 缺失。
- **P3 小图免分块**:`sr_utils.py` 宽 ≤1536 时 `tile=0`(tile_process 在 Windows 小图上原生 stack overflow)。
- **P4 清理容错**:`_clear_models`(`layer_utils.py`)对 meta 设备的 pipeline.to('cpu') 必抛
  `Cannot copy out of meta tensor`,try/except 跳过。
- **P5 utils3d 别名 + image_uv shim**:`pano_depth_utils.py` 的 `import utils3d` 改为 try/except
  别名(utils3d_moge),并补 UniK3D 时代 `image_uv` 归一化像素网格 shim。
- **P6 faces int32**:`general_utils.py` `Vector3iVector(faces…)` 前补 `.astype(np.int32)`
  (open3d int32 容器对 int64 千万级三角形索引 AV;另需 open3d≥0.18)。
- **P7 重建前释放模型**:`demo_scenegen.py` 进 mesh 组合前置空 inpaint/SR/ZIM/GDINO(几十 GB),
  否则大世界 vstack `ArrayMemoryError`。
- **gsplat `-Wno-attributes`**(打在 site-packages,`stage_c_train.py` 自动):该 flag 被 torch 转成
  `/Wno-attributes` 后 MSVC cl 报 D8021,删掉即可(仅告警抑制)。

### 5. 配置与启动

```bat
cd /d D:\hunyuan-rpc
copy config.example.json config.json   :: 改 hyworld_repo / python / work_root / token / hf_home
python hunyuan_worker.py               :: 监听 :8310
curl -H "X-Worker-Token: <token>" http://localhost:8310/health
```

`ok: true` 即就绪(python / hyworld_repo / gpu.cuda / 16 项 import 逐项打勾)。

### 6. 常驻与网络

- **开机自启**:任务计划程序新建基本任务 → 触发器「登录时」→ 操作
  `C:\Users\<你>\miniconda3\envs\hyworld1\pythonw.exe D:\hunyuan-rpc\hunyuan_worker.py`,起始于 `D:\hunyuan-rpc`。
- **防火墙**:放行 TCP 8310(作用域限定内网子网)。**不要暴露公网**,鉴权只有共享 token。
- 任务工作区 `work_root/jobs/<id>/` 完成/失败 7 天自动清理(`job_ttl_days`)。

### 7. VideoAgents 侧接线

控制台「🎨 生成模型」→「🌍 世界模型」→ **HunyuanWorld(远程)** 标签页:填 worker 地址
(`http://<GPU主机内网IP>:8310`)与 token → 「验证」显示 GPU/磁盘即通。保存后场景预览页
「➕ 生成世界模型」即走本地算力。

## 故障表

| 症状 | 处理 |
|---|---|
| worker 自检 `hyworld_repo.ok=false` | 该目录下需有 `demo_scenegen.py`:先 clone,或改正路径(注意多套一层目录的情况) |
| `Failed to import diffusers.loaders.unet ... name 'nn' is not defined` | transformers 太新:`pip install transformers==4.51.0 diffusers==0.34.0` |
| `import basicsr` 报 `No module named 'torchvision.transforms.functional_tensor'` | torchvision 0.18+ 改名 `_functional_tensor`:按第 2 节的 shim 命令补兼容文件 |
| `import zim_anything` 报 `No module named 'easydict'` | `pip install easydict` |
| 深度预测阶段 `AttributeError: module 'utils3d' has no attribute 'numpy'` | 装了 PyPI 的 `utils3d`(没有 numpy 子模块):`pip uninstall -y utils3d` 后装 `utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git` |
| `UnicodeEncodeError: 'gbk' codec can't encode character`(打印 emoji 处) | Windows 控制台默认 GBK:worker 已对阶段子进程注入 `PYTHONUTF8=1` |
| FP8 量化阶段无声崩溃(exit -1,access violation) | AngelSlim FP8 在 Windows torch 2.5 的原生稳定性问题:**不要走 FP8**,P1 补丁后 bf16 直跑 |
| 模型加载阶段无声崩溃(exit -1,无 traceback) | commit 内存上限被杀:任务管理器看「已提交 xx/yy」,yy 应 ≈140GB;小了就按「自动管理取消 → D 盘自定义 40960/81920 → **点「设置」** → 重启」加大 pagefile |
| 天空阶段 `FileNotFoundError: full_image_sr.png` | RealESRGAN 超分 OOM 被吞:P2 补丁(tile=512)由 worker 自动打 |
| inpaint/VAE 阶段 `torch.OutOfMemoryError`(报十几 GB 空闲却分配不出) | 显存碎片化;**输入全景建议 ≤2048×1024**(24G 上限),冒烟用 1024×512 |
| 渲染全黑 / 深度全 inf | open3d OffscreenRenderer(Filament)在部分 Windows 主机不可用:Stage B 已改 CPU 光线投射;另需 open3d≥0.18 |
| 生成成功但视窗里世界糊成一片/颜色形状全错 | SPZ 段落顺序必须与官方 serializePackedGaussians 一致(positions → alphas → colors → scales → rotations)。`spz_writer.py` 2026-10-20 前版本按错顺序写,更新后重跑 gs 阶段 |
| gsplat 报 `where cl` 失败 | 没编译缓存或没装编译环境:按第 2 节 gsplat 段(x64 Native Tools / vcvars64)编译一次入缓存 |
| gsplat 编译报 `cl: error D8021 "/Wno-attributes"` | gsplat 自带 flag 与 MSVC 不兼容:stage_c 启动时自动删,或手删 `gsplat\cuda\_backend.py` 里的 `-Wno-attributes` |
| `PytorchStreamReader failed reading zip archive` | 权重缓存损坏:删掉重下(超分默认 scale=2 用 `RealESRGAN_x2plus.pth`),并用 torch.load 验证可读 |
| `OSError: ... os error 112`(磁盘空间不足) | HF 缓存默认在 C 盘:`config.json` 加 `"hf_home": "D:/hf_cache"` 并重启 worker |
| `GatedRepoError: 401/403` | FLUX 是 gated 仓库:浏览器接受许可 + `huggingface-cli login`(见第 3 节) |
| worker 忙(409) | 单卡单任务;等当前任务完,或 `DELETE /worlds/<id>` 取消 |

## License

- 本仓库代码由 AgenticsWorld 发布;许可见仓库 LICENSE 文件。
- HunyuanWorld-1.0 权重与代码适用 **TENCENT HUNYUANWORLD-1.0 COMMUNITY LICENSE AGREEMENT**:
  **不适用欧盟/英国/韩国**;面向公众的产品/服务**上月活 >100 万**须向腾讯单独申请
  (hunyuan3d@tencent.com);分发衍生作品须附协议文本;服务须披露实际提供方、不得暗示腾讯背书。
  本仓库不分发模型权重——用户在自己主机自行部署。
- SPZ 导出为按公开格式(nianticlabs/spz, MIT)自研的纯 Python 编码器。
