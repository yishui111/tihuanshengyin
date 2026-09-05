# 项目约定（tihuanshengyin）

> 本文件是给 AI/协作方看的项目工作规则，随仓库分发；改代码请同步更新
> `README.md` / `DEPLOY.md` / `部署方案.md`，保持三者与代码一致。

## 项目概述

Windows 离线语音克隆/换声服务集合（模型本地加载，无需联网）：
- 工作台（hub，8000）：**一键换音色（训练音色，核心功能）**、文件夹批量说话人换声、单文件换声、输出浏览
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
   少数说话人需 ≥2 段或占比 ≥25% 才判为第二人；最多按 4 人聚类；仅 2 段时
   直接比声纹相似度（阈值 0.45：实测同人 ≈0.67 / 异人 ≈0.25）。
   VAD 的 silence_gap 按"连续静音时长"切段（换气不切碎句子）。
4. 逐段换声（hub/pipeline.py）：按说话人把语音段送 A/C 引擎 `/convert`
   （普通角色传 `auto_pitch=true`；训练音色传 `auto_pitch=false`），
   **按说话人分组连续转换**（同人所有段一次处理完，减少引擎模型切换），
   按原时间轴拼回（段边界 45ms 淡入淡出）；视频用 ffmpeg `-c:v copy -c:a aac`
   混流，画面不重编码。D 不用于批量逐段（时长不保）。
   换声粒度：单说话人整段一次转换；多说话人先合并相邻同人段（间隔 ≤1.5s）再转换。
5. hub 的 C 角色名换算：模型文件名（`nahida41_G_*.pth`/`randenEi_G_*.pth`）≠
   接口角色名（`nahida`/`raiden`），见 `hub/roles.py` 的 `C_CHAR_OVERRIDES`。
6. 各服务以脚本方式运行：若用嵌入版 Python 需确保脚本目录在 sys.path
   （`hub/server.py` 开头显式 `sys.path.insert(0, ...)`，其余服务靠 `os.chdir`+PATH）。
7. 引擎目录一律通过环境变量可覆盖：`RVC_ROOT` / `SOVITS_SRC` / `GSV_ROOT` /
   `OV_CKPT` / `TMP_ROOT` / `API_PORT` / `HUB_PORT`；脚本优先 `runtime\pyXXX\python.exe`。
8. **训练音色**（核心）：训练中心（换声模式）交付包 `交付模型\rvc\<角色>\` 整个文件夹
   复制到 `rvc_service\models\<角色>\` 即自动识别（模仿文字驱动项目 tts_api 的目录扫描，
   刷新即出现，无需注册）；hub 角色表 `id="T:<角色>"`、带 `trained=True`。
   训练音色**只换音色，禁止变调**：A 引擎侧 `discover_trained()` 识别后强制 `f0_up_key=0`
   忽略 auto_pitch（`get_vc` 靠 `weight_root` env 找模型，`_rvc_lock` 内临时翻转）；
   hub 侧 `/api/swap`（一键换声）与 `pipeline._convert_segment` 对 trained 角色传
   `auto_pitch=false`。hub→引擎的 HTTP 调用依赖 `no_proxy`（server.py 开头 setdefault）。
9. **多人对话换声**（核心流程，卡片②）：`POST /api/swap_upload`（上传原视频/音频 →
   排查说话人：几个人/各说多久/每人一段试听）→ 页面逐人分配训练音色 →
   `POST /api/swap_multi`（按说话人分组替换，输出文件名用原上传名）。不做人声分离。
10. **模型常驻显存管理**（RVC 服务）：同一模型连续转换不重载（`_current_model` 判断）；
    换模型时 `_model_unload_locked()` 先清推理图缓存并把 net_g/pipeline 置 None 再
    empty_cache——**禁用 `vc.get_vc("")`**（它 delattr 属性，下次 `if self.net_g is not
    None` 会 AttributeError）。hub 在换声任务结束后调引擎 `/model/unload`（A/C 都有），
    显存里同时只有一个音色模型、任务结束即清空。

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
---
### 关键点（2026-09-02 上传整理补充）
- 自研层：hub 工作台 8000 + 4 个引擎封装 API：A=RVC 8010 / B=OpenVoice 8020 / C=SoVITS 8030 / D=GPT-SoVITS 8040（端口/引擎目录/解释器全部环境变量可覆盖）
- 引擎源码与权重一律不入库；DEPLOY.md 记录固定 commit：RVC 81eed5e8f、GPT-SoVITS d523079f、OpenVoice 74a1d147、so-vits 4.1-Stable
- 关键坑：功能 C 需 onnxruntime-gpu==1.17.1 + nvidia-cudnn-cu11（已写进 requirements）；B/工作台可 CPU，C/D 需 NVIDIA
- requirements 除 onnxruntime 外多为宽松版本，torch 以各引擎 README 为准
- 角色音色权重属授权资产不入库；stop.bat 保留 pause（双击体验）
