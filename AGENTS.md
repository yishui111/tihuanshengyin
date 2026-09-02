# 项目约定（tihuanshengyin）

> 本文件是给 AI/协作方看的项目工作规则，随仓库分发；改代码请同步更新
> `README.md` / `DEPLOY.md` / `部署方案.md`，保持三者与代码一致。

## 项目概述

Windows 离线语音克隆/换声服务集合（模型本地加载，无需联网）：
- 工作台（hub，8000）：文件夹批量说话人换声、单文件换声、输出浏览
- 功能A RVC 二次元角色换声（8010）
- 功能B OpenVoice 任意人声克隆（8020）
- 功能C SoVITS 中配角色换声（8030）
- 功能D GPT-SoVITS ASR+重合成（8040）

## 仓库边界（重要）

- 本仓库**只含自研代码**：`hub\`（编排层）、四个 `*_service\*.py`（引擎封装 API）、脚本与文档。
- **不入库**（.gitignore 已忽略，部署时才出现）：第三方引擎源码
  （`rvc\`、`sovits_service\so-vits-svc-4.1-Stable\`、`gptsovits_service\GPT-SoVITS\`）、
  `runtime\`（Python 运行时 + ffmpeg + 模型缓存）、各服务权重目录（`models\`、
  `checkpoints_v2\`、`asr\`、`pymss_models\`）、素材 `ziliao\`。
- 新机器按 `DEPLOY.md` 装引擎/权重/运行时后，代码零修改即可启动。

## 目录/代码速查

- `hub\server.py`（FastAPI 编排）→ `hub\diarize.py`（说话人检测）→
  `hub\pipeline.py`（批量管线）→ `hub\roles.py`（角色注册表）；
  `hub\roles.py` 的 `PORTS`/`CN_NAMES`/`C_CHAR_OVERRIDES` 是角色与端口的唯一事实来源之一。
- 四个封装 API 头部注释写明了各自依赖的引擎目录、模型放置规则与环境变量。
- 启停脚本：`start.bat`（支持 `A/B/C/D`、`stop`、`help` 参数）/ `stop.bat`（全部停止）。
- 自检：`GET /api/health`（hub）；`GET /health`（各引擎）；`POST /convert` 冒烟。

## 关键代码约定（非显而易见，改动时勿破坏）

1. 功能B：素材写 `input.wav`、参考写 `ref.wav`，**禁止共用同一个文件名**
   （曾因共用 `normalized.wav` 导致素材被参考覆盖）。
2. 功能B：音色向量用能量 VAD 分段 + 多段平均（`extract_se_robust`），
   不引入需联网的 silero VAD；默认 `tau=0.15`。
3. 说话人检测（hub/diarize.py）：能量 VAD → ECAPA 声纹（speechbrain
   `spkrec-ecapa-voxceleb`，离线缓存 `runtime\cache\`）→ 余弦距离层次聚类；
   少数说话人需 ≥2 段或占比 ≥25% 才判为第二人。
4. 逐段换声（hub/pipeline.py）：按说话人把语音段送 A/C 引擎 `/convert`
   （只传 `character` + `auto_pitch=true`），按原时间轴拼回（段边界 45ms 淡入淡出）；
   视频用 ffmpeg `-c:v copy -c:a aac` 混流，画面不重编码。D 不用于批量逐段（时长不保）。
   换声粒度：单说话人整段一次转换；多说话人先合并相邻同人段（间隔 ≤1.5s）再转换。
5. hub 的 C 角色名换算：模型文件名（`nahida41_G_*.pth`/`randenEi_G_*.pth`）≠
   接口角色名（`nahida`/`raiden`），见 `hub/roles.py` 的 `C_CHAR_OVERRIDES`。
6. 各服务以脚本方式运行：若用嵌入版 Python 需确保脚本目录在 sys.path
   （`hub/server.py` 开头显式 `sys.path.insert(0, ...)`，其余服务靠 `os.chdir`+PATH）。
7. 引擎目录一律通过环境变量可覆盖：`RVC_ROOT` / `SOVITS_SRC` / `GSV_ROOT` /
   `OV_CKPT` / `TMP_ROOT` / `API_PORT` / `HUB_PORT`；脚本优先 `runtime\pyXXX\python.exe`。

## 已知问题（勿当新 bug 报）

- onnxruntime 必须 `==1.17.1`（CUDA 11.8）+ `nvidia-cudnn-cu11`，否则 SoVITS onnx 回退 CPU 极慢。
- 两人重叠说话无法完美分离；功能D 会把句子"重新说一遍"，保留原节奏用功能C。
- 端口被占用：`set API_PORT=xxxx` / `set HUB_PORT=xxxx` 后重启。

## 禁止（Do NOT）

- 不把真人素材、他人角色权重、训练文本、密钥提交进仓库。
- 不修改/删除他人部署目录里的引擎与权重（`.gitignore` 目录不提交即可）。
- 不宣称完成而未跑通自检（health + 一次 /convert 冒烟）。

## 沟通与交付

- 默认简体中文；改动后同步更新 README/DEPLOY/部署方案；保持本文件 <150 行。
