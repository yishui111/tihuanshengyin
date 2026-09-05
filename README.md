<div align="center">

# 🎙️ 多引擎离线声音替换工作台（换声 / 变声）

> ⭐ **喜欢这个项目？请先点个 Star 支持一下，让更多人看到！** ⭐

![GitHub stars](https://img.shields.io/github/stars/yishui111/tihuanshengyin.svg?style=flat-square&color=orange)
![GitHub forks](https://img.shields.io/github/forks/yishui111/tihuanshengyin.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/yishui111/tihuanshengyin.svg?style=flat-square)

**一套自研的多引擎「声音替换」服务集合：文件夹批量换声工作台 + RVC / OpenVoice / SoVITS / GPT-SoVITS 四种引擎封装 API。给一段音频/视频，自动分析说话人、按人分配角色、逐段换声并混流回视频。**

</div>

---

## ✨ 项目简介

本项目把四种主流开源声音转换引擎统一封装成一套 **Windows 离线**声音替换工具：

- 一个**统一工作台（hub）**做批量编排：扫描文件夹 → 自动检测每个音/视频里有几个说话人（VAD + 声纹聚类）→ 为每个说话人指定角色 → 自动逐段换声 → 视频画面不变、音轨替换后混流出 mp4；
- 四个**独立引擎服务**（各自 FastAPI + 网页小界面）：RVC 二次元角色换声、OpenVoice 任意人声零样本克隆、SoVITS 中配角色换声、GPT-SoVITS「ASR 识别 + 重合成」。

适合：视频/播客二创换音色、角色对话配音、把真人素材转成动漫角色音色等离线处理场景。
数据不出本机，模型全部本地加载，无需联网（除首次按 `DEPLOY.md` 下载引擎与权重）。

> ⚠️ 本仓库只包含**自研的封装层代码 / 编排逻辑 / 文档**。
> 第三方引擎源码、模型权重、角色音色、Python 运行时等大件**不随仓库分发**，请按下方「大件资源下载」与 `DEPLOY.md` 在新机器上自行准备。

## 🎯 主要功能

| 入口 | 端口 | 能力 |
|---|---|---|
| **换声工作台**（主页面） | 8000 | **一键换音色（训练音色，核心）**、多人对话换声（排查说话人逐人分配）、文件夹批量说话人换声、输出浏览/试听/下载/删除 |
| **训练音色**（一键换声） | 8010 | 原视频/音频 → 只换音色为训练好的真人音色，**音高/语调/时长保持原样**；模型放 `rvc_service\models\<角色>\`（训练中心交付包整个文件夹复制进来即自动识别） |
| **功能A** RVC 二次元角色换声 | 8010 | 素材 + 角色名 → 换成该角色音色（保留语速/停顿/情绪），角色 .pth 放 `rvc\assets\weights\` |
| **功能B** OpenVoice 任意人声克隆 | 8020 | 素材 + 一段 5~30 秒参考人声 → 克隆成参考人的音色，**无需训练** |
| **功能C** SoVITS 中配角色换声 | 8030 | 中配角色，自然度比 RVC 高（男声转萝莉音更自然），角色模型放 `sovits_service\models\` |
| **功能D** GPT-SoVITS ASR+重合成 | 8040 | 语音识别成文字后用角色音色「重新说一遍」（不用于逐段替换，时长不保） |

- 🔍 **批量说话人换声**：一人对话素材自动分成多个说话人，可为每个说话人分配**不同引擎的角色**（如 说话人1→可莉(SoVITS)、说话人2→派蒙(RVC)）；
- 🎚️ **自动音高匹配**：二次元角色按素材实际音高自动对准角色目标音高（变调半音数自动计算）；**训练音色则强制不变调**（只换音色，音高/语调/时长保持原样）；
- 🧩 **超长音频自动分段**：>90 秒自动分段转换再拼接，电影/长录音也能处理；
- 🎬 **视频混流**：`ffmpeg -c:v copy -c:a aac`，画面不重编码、只替换音轨；
- 💻 **四个引擎独立部署**：可只开需要的服务（省显存），端口可用环境变量覆盖。

## 🗂️ 目录结构

```
tihuanshengyin/
├── hub/                        # 换声工作台（8000，自研编排层）
│   ├── server.py               # FastAPI 主服务 + 批量任务调度
│   ├── diarize.py              # VAD + ECAPA 声纹聚类 → 说话人检测
│   ├── pipeline.py             # 批量管线：分离 → 逐段换声 → 拼接 → 混流
│   ├── roles.py                # 角色注册表（A/C/D 合并、在线状态）
│   └── web/index.html          # 工作台页面
├── openvoice_service/          # 功能B 服务（8020，自研封装）
│   └── openvoice_clone_api.py  # OpenVoice 任意人声克隆 API
├── rvc_service/                # 功能A 服务（8010，自研封装）
│   ├── rvc_character_api.py    # RVC 角色换声 API（含训练音色目录扫描）
│   └── models/                 # 训练音色模型（训练中心交付包复制到这里，自动识别）
├── sovits_service/             # 功能C 服务（8030，自研封装）
│   └── sovits_cn_api.py        # SoVITS 中配角色换声 API
├── gptsovits_service/          # 功能D 服务（8040，自研封装）
│   └── gptsovits_cn_api.py     # GPT-SoVITS ASR+重合成 API
├── requirements-py310.txt      # 功能B 封装层依赖（py310）
├── requirements-py312.txt      # 工作台 + A/C/D 封装层依赖（py312）
├── start.bat / stop.bat        # 一键启停（支持按引擎启停）
├── AGENTS.md                   # 项目约定（AI/协作工作规则）
├── 部署方案.md                  # 复原清单与技术约定（活文档）
└── DEPLOY.md                   # 新机器完整部署手册
```

> 💡 说明：引擎源码与运行时在部署后才出现且被 `.gitignore` 忽略：
> `rvc\`（RVC-WebUI 克隆）、`sovits_service\so-vits-svc-4.1-Stable\`、
> `gptsovits_service\GPT-SoVITS\`、`runtime\`（Python 运行时 + ffmpeg + 模型缓存）、
> 各服务 `models\`/`checkpoints_v2\`（权重）等，详见 `DEPLOY.md`。

## 🚀 快速开始

> 完整的分步部署（装引擎 → 下权重 → 放目录）请看 **[DEPLOY.md](DEPLOY.md)**；
> 下面假设引擎与模型已按 DEPLOY 就位。

### 环境要求

- Windows 10/11 64 位（本套脚本为 `.bat`）；
- NVIDIA 显卡（驱动 ≥ 522）：功能C（SoVITS）与功能D（GPT-SoVITS）推理走 CUDA，**必须有**；功能B 与工作台可 CPU 运行（较慢），功能A 建议 GPU；
- 磁盘剩余 ≥ 40GB（运行时 + 各引擎 + 权重）；内存 ≥ 16GB 建议；
- Python 3.10（功能B）/ 3.12（工作台 + A/C/D），由 `runtime\py310|py312` 提供（见 DEPLOY）。

### 1. 克隆

```bash
git clone https://github.com/yishui111/tihuanshengyin.git
cd tihuanshengyin
```

### 2. 准备引擎与权重（一次性的"重活"，按 DEPLOY.md 执行）

各引擎源码放置位置、固定 commit、预训练资产下载与本地目标路径、角色权重如何自制/拷贝，全部在 **[DEPLOY.md](DEPLOY.md)** 中逐步列出。

### 3. 启动

双击 **`start.bat`**（默认启动 工作台 + 功能A），或命令行：

```
start.bat           启动 工作台 + 功能A（RVC，默认）
start.bat A C       再加开功能C（SoVITS 中配角色）
start.bat B         只开 工作台 + 功能B（OpenVoice 克隆）
start.bat A B C D   全部引擎都开
start.bat stop      停止全部服务
start.bat stop B    只停止功能B（释放显存）
start.bat help      查看用法与引擎说明
```

- 脚本自动跳过已运行的服务（不会重复启动、不会端口冲突）；
- 启动后自动打开工作台页面 `http://127.0.0.1:8000/`。

### 4. 验证

- 打开浏览器访问 http://127.0.0.1:8000/ ，页面顶部能看到各引擎在线状态（绿点 = 在线）；
- 命令行冒烟：`curl http://127.0.0.1:8000/api/health`、`curl http://127.0.0.1:8030/health` 均返回 200。

## 📥 大件资源下载（引擎 / 模型 / 运行时，不入库）

| 资源 | 用途 | 下载地址 / 获取方式 |
| ---- | ---- | ---- |
| RVC-WebUI 引擎（`rvc\`） | 功能A 推理引擎 | https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI （固定 commit 见 DEPLOY.md） |
| so-vits-svc 引擎（`sovits_service\so-vits-svc-4.1-Stable\`） | 功能C 推理引擎 | https://github.com/svc-develop-team/so-vits-svc （4.1-Stable 分支） |
| GPT-SoVITS 引擎（`gptsovits_service\GPT-SoVITS\`） | 功能D 推理引擎 | https://github.com/RVC-Boss/GPT-SoVITS （固定 commit 见 DEPLOY.md） |
| OpenVoice 引擎源码（pip 安装用） | 功能B 依赖包 | https://github.com/myshell-ai/OpenVoice （`pip install -e`，见 DEPLOY.md） |
| OpenVoice V2 checkpoint（`openvoice_service\checkpoints_v2\`） | 功能B 音色转换模型 | 官方 README 内 V2 权重链接（converter/config.json + checkpoint.pth） |
| SenseVoice ASR（`gptsovits_service\asr\SenseVoiceSmall\`） | 功能D 语音识别 | https://huggingface.co/FunAudioLLM/SenseVoiceSmall |
| GPT-SoVITS 预训练（hubert/roberta/s1/s2 等） | 功能D | 见 GPT-SoVITS 官方安装说明 |
| RVC 预训练（hubert_base / rmvpe） | 功能A | 见 RVC-WebUI 官方安装说明 |
| SoVITS 预训练（vec-768/vec-256 ONNX、rmvpe.pt） | 功能C | 见 so-vits-svc 官方安装说明 |
| ECAPA 声纹模型（speechbrain `spkrec-ecapa-voxceleb`） | 工作台说话人检测 | 首次运行自动下载到 `runtime\cache\`（离线放置法见 DEPLOY.md） |
| pymss 人声分离模型 `bs_roformer_voc_hyperacev2` | 工作台人声分离 | https://github.com/pymss-project/pymss （放入 `rvc_service\pymss_models\`） |
| ffmpeg / ffprobe | 音频提取与视频混流 | https://ffmpeg.org/ （放入 `runtime\ffmpeg\bin\`） |
| 角色音色权重（A/C/D） | 各引擎角色 | ⚠️ **自备**：本仓库不含任何角色权重与真人素材；用你自己的训练工程产出，或从你原来的部署拷贝（结构见 DEPLOY.md） |

> 角色音色通常来自配音演员/特定音色训练，涉及授权问题，**不要**上传他人训练的角色模型。

## 🛠️ 本地开发 & 提交

```bash
git add .
git commit -m "feat: xxx"
git push origin main
```

代码里引擎目录、端口全部通过环境变量/常量可覆盖（见各服务文件头部注释与 DEPLOY.md「配置差异项」），换机器不需要改代码。

## ❓ 常见问题（FAQ）

- **Q：为什么仓库里没有 `rvc\`、`runtime\`、`models\`？** A：它们是第三方引擎 / 运行时 / 权重，体积大且非本项目自研，按「铁律」不入库；请按 `DEPLOY.md` 在新机器克隆引擎、下载权重到对应目录，`start.bat` 会自动找到它们。
- **Q：`start.bat` 双击后服务窗口一闪而过？** A：通常是杀毒软件拦截 python.exe，把 `runtime\pyXXX\python.exe` 加入白名单后重试。
- **Q：页面提示某引擎离线？** A：对应功能服务没启动或端口被占用，先 `start.bat A` 之类启动它；工作台顶部状态条可确认。
- **Q：功能C 首次加载很慢 / 像卡死？** A：SoVITS 首个角色加载约 15~25 秒属正常；若 onnx 一直回退 CPU（首次加载 80 秒+），请确认装了 `onnxruntime-gpu==1.17.1`（CUDA 11.8 版），见 DEPLOY.md「常见问题」。
- **Q：说话人检测不准？** A：两人重叠说话无法完美分离；素材尽量清晰、单人/双人分开录。
- **Q：无显卡能跑吗？** A：功能C/D 的推理代码走 CUDA，需要 NVIDIA 显卡；功能B 与工作台可 CPU（慢，10 秒音频约几十秒）；功能A 视 RVC 引擎构建（CUDA 版默认 GPU）。
- **Q：原项目里的 `tests\` 去哪了？** A：原 `tests\` 主要是本机自测用的音频素材（含真人录音），属于个人素材不入库；部署自检方法见 DEPLOY.md「步骤 5」，不需要额外测试文件。
- **Q：可以商用吗？** A：请遵守各引擎开源协议以及音色/素材的授权要求；本项目仅供学习交流。

## ⚠️ 注意事项

- 敏感信息（密钥、token、账号密码）一律放环境变量或 `.env`，禁止提交到仓库；
- 请勿把真人素材、他人角色权重、训练文本上传到公开仓库；
- 使用他人声音/角色进行创作时请尊重授权与相关法律法规；
- 本仓库仅供学习交流使用。

## 📄 许可证

MIT License（第三方引擎与模型遵循各自的开源协议）。

## 🙏 支持与致谢

如果这个项目帮到了你，**请点亮右上角的 ⭐ Star**，你的支持是我持续更新的最大动力！
