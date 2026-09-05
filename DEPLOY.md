
## 🚀 换电脑部署（保证可用）

> **方式 A（推荐 · 100% 保证）**：用 U 盘 / 网盘把「原项目整份文件夹」（含全部大件）复制到新电脑 → 双击 `start.bat` 即可。
>
> **方式 B（代码装配）**：`git clone` 本仓库 → 双击 `assemble.bat` 预检大件 → 按提示补齐缺失项（下载地址见下文/README）→ 双击 `start.bat`。

> 说明：引擎、模型、镜像、运行时等大件体积超过 GitHub 单文件 100MB 上限，**不随仓库分发**；本仓库承载全部自研代码与装配指引，"方式 A"是换机部署最稳路径，"方式 B"适合需要重新下载大件的场景。
# DEPLOY.md · 新机器部署手册

> 本仓库只含**自研代码/脚本/文档**。要在新电脑跑起来，需要按本手册：
> ① 装 Python 运行时 → ② 克隆/放置 4 个开源引擎源码 → ③ 下载预训练资产与模型权重 → ④ 放角色模型 → ⑤ `start.bat` 启动并自检。
> 目录结构与代码约定的权威速查另见同目录 `部署方案.md`。

---

## 一、部署后的最终目录形态（对照核对用）

```
<repo root>\
├─ hub\                            # ★自研：换声工作台（端口 8000）
├─ openvoice_service\              # ★自研：功能B API（8020）
│   └─ checkpoints_v2\             #    ← OpenVoice V2 权重（下载）
├─ rvc_service\                    # ★自研：功能A API（8010）
│   └─ pymss_models\vocal\...      #    ← 人声分离模型（下载，工作台分离用）
├─ rvc\                            # ← 引擎①：RVC-WebUI 克隆
│   └─ assets\{hubert_base,rmvpe,weights,indices}
├─ sovits_service\                 # ★自研：功能C API（8030）
│   ├─ so-vits-svc-4.1-Stable\     #    ← 引擎②：so-vits-svc（4.1-Stable）源码
│   │   └─ pretrain\               #    ← vec-*.onnx / rmvpe.pt 等预训练
│   └─ models\                     #    ← 角色模型 json+pth+index（自备）
├─ gptsovits_service\              # ★自研：功能D API（8040）
│   ├─ GPT-SoVITS\                 #    ← 引擎③：GPT-SoVITS 克隆
│   ├─ asr\SenseVoiceSmall\        #    ← 中文 ASR 模型（下载）
│   └─ models\<角色>\              #    ← 角色权重（自备）
├─ runtime\                        # ← Python 运行时 + ffmpeg + 模型缓存（自建）
│   ├─ py310\                      #    Python 3.10 环境（功能B 用）
│   ├─ py312\                      #    Python 3.12 环境（工作台+A/C/D 用）
│   ├─ ffmpeg\bin\{ffmpeg,ffprobe}.exe
│   └─ cache\{hf_speaker_model,huggingface,torch}
├─ start.bat / stop.bat
└─ README.md / DEPLOY.md / 部署方案.md / AGENTS.md
```

> `runtime\`、各引擎源码目录、`models\`、`checkpoints_v2\` 等都被 `.gitignore` 忽略，
> 不会被提交/覆盖；`git pull` 不会影响它们。

---

## 二、环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11 64 位 |
| 显卡 | 推荐 NVIDIA（驱动 ≥ 522）；**功能C（SoVITS）与功能D（GPT-SoVITS）的推理代码使用 CUDA，必须 NVIDIA 显卡**；功能B 与工作台可 CPU（慢） |
| 磁盘 | 剩余 ≥ 40GB（Python 环境 + 引擎 + 权重约 25~35GB） |
| 内存 | ≥ 16GB（多引擎同时开建议 32GB） |
| 网络 | 首次部署需联网下载引擎/权重（之后完全离线运行） |
| Python | 3.10 与 3.12（64 位），见下节 |
| 杀毒软件 | 需放行 `runtime\py312\python.exe` / `runtime\py310\python.exe`，否则服务窗口一闪而过 |

---

## 三、步骤 1：安装 Python 与 ffmpeg

### 3.1 安装 Python 3.10 / 3.12

从 https://www.python.org/downloads/windows/ 安装 64 位 **3.10** 与 **3.12**（勾选 Add to PATH），
或使用 conda/miniconda（`conda create -n py310 python=3.10` / `-n py312 python=3.12`）。

### 3.2 创建项目内 venv（脚本按此目录找解释器）

```bat
cd <仓库根目录>
py -3.10 -m venv runtime\py310
py -3.12 -m venv runtime\py312
```

> `start.bat` 查找解释器的顺序：环境变量 `PY310_PYTHON`/`PY312_PYTHON` →
> `runtime\pyXXX\python.exe`（便携/嵌入版布局）→ `runtime\pyXXX\Scripts\python.exe`（venv 布局）。
> 若你的环境不是上述两种，先 `set PY312_PYTHON=你的python.exe路径` 再启动即可。

### 3.3 安装 ffmpeg

下载 Windows 构建（https://ffmpeg.org/download.html 或 https://www.gyan.dev/ffmpeg/builds/），
把 `ffmpeg.exe`、`ffprobe.exe` 放到 `runtime\ffmpeg\bin\`（代码与脚本默认该路径）。

### 3.4 创建缓存目录

```bat
mkdir runtime\cache\hf_speaker_model runtime\cache\huggingface runtime\cache\torch
```

---

## 四、步骤 2：放置 4 个开源引擎（都是第三方源码，不入库）

| 引擎 | 官方地址 | 放到 | 说明 |
|---|---|---|---|
| ① RVC-WebUI | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI | `rvc\` | `git clone` 后 `git checkout 81eed5e8f68b6bed1789f682fe78cdd324495afc`（功能A 封装代码按该布局 `infer\vc\modules.py` 导入） |
| ② so-vits-svc | https://github.com/svc-develop-team/so-vits-svc | `sovits_service\so-vits-svc-4.1-Stable\` | 下载 **4.1-Stable** 分支源码解压到该目录（功能C 从该目录 `inference.infer_tool` 导入） |
| ③ GPT-SoVITS | https://github.com/RVC-Boss/GPT-SoVITS | `gptsovits_service\GPT-SoVITS\` | `git clone` 后 `git checkout d523079fc05d9a8028d6085bffe4a2757c32abb6`（功能D 导入其根目录 `api.py`） |
| ④ OpenVoice（功能B 依赖包） | https://github.com/myshell-ai/OpenVoice | 任意目录（如 `openvoice_src\`，已在 .gitignore） | `git clone`（固定 commit `74a1d147b17a8c3092dd5430504bd83ef6c7eb23`）后在 py310 环境 `pip install -e <该目录>`，使 `openvoice` 包可导入 |

> 引擎自身依赖按各引擎官方 README 安装（如 RVC 的 `requirments_cu118_py312.txt`、
> GPT-SoVITS 的 `requirements.txt` + 官方预训练下载脚本）。只做推理不需要装训练依赖，按官方指引最小化安装即可。

---

## 五、步骤 3：安装 Python 依赖

### 5.1 py312 环境（工作台 + 功能A/C/D）

```bat
runtime\py312\Scripts\python.exe -m pip install --upgrade pip
:: 先装 CUDA 版 torch（GPU）——注意 11.8 轮子与 onnxruntime 匹配
runtime\py312\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
runtime\py312\Scripts\python.exe -m pip install -r requirements-py312.txt
```

> 无 NVIDIA 显卡时把第一行换成 CPU 版：`pip install torch torchaudio`（默认源即可）；
> 但功能C/D 仍需 CUDA，无卡机器只能运行 工作台/功能A(B) 相关能力。

`requirements-py312.txt` 内容（工作台/封装层依赖，引擎自身的重型依赖已在第 4 步按引擎 README 装好）。
**重要版本坑**：onnxruntime 必须 `==1.17.1`（CUDA 11.8 版，配合 `nvidia-cudnn-cu11` 提供 cudnn64_8.dll）；
升到 1.18+ 会因缺少 CUDA 12 dll 使 onnx 回退 CPU，SoVITS 首次加载 80 秒+（像卡死）。
功能C 代码启动时会自动把 `runtime\py312\Lib\site-packages\nvidia\*\bin` 与 torch dll 目录注入 PATH。

### 5.2 py310 环境（功能B，OpenVoice）

```bat
runtime\py310\Scripts\python.exe -m pip install --upgrade pip
runtime\py310\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
runtime\py310\Scripts\python.exe -m pip install -e <OpenVoice克隆目录>
runtime\py310\Scripts\python.exe -m pip install -r requirements-py310.txt
```

> OpenVoice V2 官方环境为 torch 2.1.2 + cu118（其 README 有对应安装命令）。
> 水印库 wavmark 首次需联网下载模型；离线时功能B 代码会自动降级为无水印模式，不影响换声。

---

## 六、步骤 4：下载预训练资产与模型权重（放置路径必须精确）

### 6.1 功能A（RVC）— `rvc\assets\`

| 文件 | 目标路径 | 来源 |
|---|---|---|
| hubert 底模 | `rvc\assets\hubert_base\pytorch_model.bin` + `config.json` 等 | 按 RVC-WebUI 官方「预训练模型下载」（hubert_base） |
| f0 提取 | `rvc\assets\rmvpe\rmvpe.pt` | 同上（rmvpe） |
| 角色模型 | `rvc\assets\weights\<角色名>.pth` | ⚠️ 自备（训练产出或原部署拷贝） |
| 角色索引（可选） | `rvc\assets\indices\<角色名>.index` | 自备，能提高相似度 |

> 工作台会扫描 `weights\*.pth` 自动把角色加入下拉列表（`hub\roles.py`）。

### 6.1b 训练音色（训练中心交付包）— `rvc_service\models\`

| 文件 | 目标路径 | 来源 |
|---|---|---|
| 训练音色交付包 | `rvc_service\models\<角色名>\`（含 `<角色名>.pth` + `<角色名>.index`） | 训练中心（`启动训练中心-换声.bat`）训练完生成 `交付模型\rvc\<角色名>\`，**整个文件夹复制过来** |

> 目录扫描自动识别，复制后刷新工作台即出现（无需注册/重启），角色 id 为 `T:<角色名>`。
> 训练音色**只换音色，不改变音高/语调/时长**（引擎强制 `f0_up_key=0`，忽略 auto_pitch）；
> 工作台首页「一键换音色」卡片即用它处理原视频/原音频。

### 6.2 功能C（SoVITS）— `sovits_service\`

预训练放 `sovits_service\so-vits-svc-4.1-Stable\pretrain\`：

| 文件 | 用途 | 来源 |
|---|---|---|
| `vec-768-layer-12.onnx` | contentvec 编码器（ONNX） | so-vits-svc 官方发布（4.1 默认角色用） |
| `vec-256-layer-9.onnx` | contentvec 编码器（ONNX） | 同上（旧 4.0 角色用，按角色 json 的 `speech_encoder` 决定） |
| `rmvpe.pt` | f0 音高提取 | so-vits-svc 官方发布 |

角色模型放 `sovits_service\models\`，**每个角色一组文件**，并以 json 的 `speech_encoder` 对齐上面的预训练：

```
models\nahida41_G_111200.pth       主模型
models\nahida41.json               配置（含 speech_encoder: vec768l12-onnx、spk 名）
models\nahida41_feature_and_index.pkl  特征检索索引（4.1 特征检索角色）
models\klee_G.pth / klee.json / klee_kmeans.pt   （kmeans 索引角色）
```

注册规则（踩坑必读）：
- `sovits_service\sovits_cn_api.py` 顶部 `CHARACTERS` 登记接口角色名（键）+ 文件名 + 目标音高；
- json **文件名** = 引擎接口名（如 `klee.json`），但模型文件命名可能不同，如 `nahida41_G_*.pth` ↔ 接口名 `nahida`、`randenEi_G_*.pth` ↔ `raiden`，需在 `hub\roles.py` 的 `C_CHAR_OVERRIDES` 做文件名→接口名换算；
- 下载不到的角色权重请用你自己的训练工程产出（或从原部署整目录拷贝 `models\`）。

### 6.3 功能D（GPT-SoVITS）— `gptsovits_service\`

| 文件/目录 | 目标路径 | 来源 |
|---|---|---|
| 引擎预训练（hubert/roberta/s1/s2 等） | `GPT-SoVITS\GPT_SoVITS\pretrained_models\...` 与 `GPT-SoVITS\pretrained_models\...` | 按 GPT-SoVITS 官方安装/下载脚本 |
| 中文 ASR 模型 | `gptsovits_service\asr\SenseVoiceSmall\`（内含 model.pt/config.yaml/...） | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| 角色权重（每个角色一个目录） | `gptsovits_service\models\<角色名>\`，内含 `*.ckpt` + `*.pth` + `ref.wav` + `ref_text.txt` | ⚠️ 自备 |

> `ref.wav` 为该角色 5~10 秒参考音频、`ref_text.txt` 为其转写文本（若为空则在
> `gptsovits_cn_api.py` 的 `CHARACTERS` 中填 `ref_text`）；角色目录出现在下拉列表靠
> `roles.py` 扫描 `models\` 下含 .ckpt/.pth 的子目录。

### 6.4 功能B（OpenVoice）— `openvoice_service\`

| 文件 | 目标路径 | 来源 |
|---|---|---|
| V2 权重 | `openvoice_service\checkpoints_v2\converter\{config.json, checkpoint.pth}` | OpenVoice 官方 README 的 V2 权重链接（约 130MB） |
| 内置音色（可选） | `openvoice_service\checkpoints_v2\base_speakers\ses\*.pth` | 同上（封装代码只用自定义参考人声，可不下） |

### 6.5 工作台（hub）离线模型缓存

| 模型 | 目标 | 说明 |
|---|---|---|
| ECAPA 声纹（说话人检测） | `runtime\cache\hf_speaker_model\` + `runtime\cache\huggingface\` | 首次联网运行 `python hub\server.py` 会自动从 speechbrain（`spkrec-ecapa-voxceleb`）下载并缓存；之后离线。也可提前在能联网的机器上把缓存目录整个拷过来 |
| 人声分离 `bs_roformer_voc_hyperacev2` | `rvc_service\pymss_models\vocal\vocal_extraction\{bs_roformer_voc_hyperacev2.ckpt, bs_roformer_voc_hyperacev2.yaml}` | 来自 pymss（https://github.com/pymss-project/pymss），按官方下载到该目录；工作台批量换声勾选「人声分离」时才需要 |

---

## 七、步骤 5：启动与自检

```bat
start.bat            :: 工作台(8000) + 功能A(8010)
start.bat A B C D    :: 全开
start.bat stop       :: 全部停止（stop.bat 同）
```

自检：

```powershell
curl http://127.0.0.1:8000/api/health          # 工作台：{"status":"ok",...}
curl http://127.0.0.1:8010/health              # A
curl http://127.0.0.1:8020/health              # B
curl http://127.0.0.1:8030/health              # C
curl http://127.0.0.1:8040/health              # D

# 功能C 冒烟（需要已放置一个角色模型，如 klee）：
curl -X POST -F "audio=@<你的测试音频>.wav" -F "character=klee" http://127.0.0.1:8030/convert -o out.wav
```

判定：接口返回 200；输出 wav 时长 ≈ 素材时长；听感音色向角色靠拢。

---

## 八、端口 / 目录 / 环境变量（配置差异项）

| 项 | 默认值 | 可覆盖 |
|---|---|---|
| 换声工作台 | http://127.0.0.1:8000/ | 环境变量 `HUB_PORT` |
| 功能A/B/C/D | 8010 / 8020 / 8030 / 8040 | 环境变量 `API_PORT`（作用于各自服务） |
| Python 解释器 | `runtime\py310\|py312\python.exe` | `PY310_PYTHON` / `PY312_PYTHON` |
| 引擎目录 | `rvc\`、`sovits_service\so-vits-svc-4.1-Stable\`、`gptsovits_service\GPT-SoVITS\` | `RVC_ROOT` / `SOVITS_SRC` / `GSV_ROOT` |
| OpenVoice 权重目录 | `openvoice_service\checkpoints_v2` | `OV_CKPT`（另 `OV_DEVICE`=cuda/cpu、`OV_WATERMARK`） |
| 输出目录 | `<仓库根>\ziliao\输出\`（批量在 `batch_时间戳\` 子目录） | 代码自动创建 |
| 临时目录 | 各服务 `tmp\`、`ziliao\tmp_batch`、`ziliao\tmp_one` | `TMP_ROOT`（各服务） |
| 服务日志 | 各服务窗口（控制台）；也可自行 `> service_x.out.log` 重定向 | — |
| 人声分离时长阈值 | 90 秒（超长自动分段） | `SEPARATE_MAX_SEC` |

代码全部用相对路径（`os.path.dirname(__file__)` 推导），换目录部署**不需要改代码**。

---

## 九、常见问题排查

| 现象 | 处理 |
|---|---|
| 服务窗口一闪而过 | 杀毒软件拦截 python.exe → 加白名单重试 |
| `[ERROR] runtime python not found` | 未建 venv 或路径不对；按第三节建好，或用 `PY312_PYTHON` 指定 |
| 页面打不开 | 看服务窗口报错；`netstat -ano \| findstr 8010` 查端口占用 |
| 提示「引擎离线」 | 对应服务没启动/端口被占；先启动它 |
| C 首次加载 80 秒+ / 像卡死 | onnx 回退 CPU → 确认 `onnxruntime-gpu==1.17.1` + `nvidia-cudnn-cu11`（见 5.1） |
| B 提示缺少 openvoice 包 | 没在 py310 里 `pip install -e` OpenVoice（见第四节④） |
| D 提示缺少 ASR/模型 | 确认 `asr\SenseVoiceSmall\` 完整、`GPT-SoVITS\GPT_SoVITS\pretrained_models\` 就位 |
| 说话人检测不准 | 重叠说话无法完美分离；素材清晰、单人/双人分开录 |
| CPU 模式很慢 | 正常（10 秒音频约几十秒）；C/D 需 NVIDIA 显卡 |
| 端口被占用 | 改对应 bat 顶部默认端口，或启动前 `set API_PORT=xxxx` / `set HUB_PORT=xxxx` |
| 显存不足（<8GB） | 别同时开 A/B/C/D，只开需要的，如 `start.bat C` |
