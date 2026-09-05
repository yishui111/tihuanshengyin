# -*- coding: utf-8 -*-
"""
服务A：RVC 二次元角色换声 API（独立部署）
=========================================
功能：上传素材音频 + 角色名，输出换成该二次元角色音色的 wav。
      保留原始语速、停顿、情绪，只改音色（f0_up_key 可额外变调）。

引擎：RVC-WebUI 当前 main 分支（Python 3.12 + torch 2.7.1+cu118）
端口：默认 8010（环境变量 API_PORT 可改）

运行：
    在仓库根目录运行（默认 RVC_ROOT = 仓库根\\rvc，可用环境变量 RVC_ROOT 覆盖）：
    runtime\\py312\\python.exe rvc_service\\rvc_character_api.py

可选环境变量：
    RVC_ROOT  RVC 仓库根目录（默认：本文件所在目录）
    API_PORT  监听端口（默认 8010）
    TMP_ROOT  临时目录（默认 <RVC_ROOT>/TEMP/rvc_api）
    UVR_MODEL 分离模型名（默认 UVR-MDX-NET-Inst_HQ_3.onnx）

角色模型放置（必读）：
    <RVC_ROOT>\\assets\\weights\\<角色名>.pth
    <RVC_ROOT>\\assets\\indices\\<角色名>.index   （可选，能提高相似度）
"""

import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid

# ---------------- 运行环境（必须在导入 RVC 模块之前完成） ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_ROOT = os.environ.get(
    "RVC_ROOT", os.path.normpath(os.path.join(SCRIPT_DIR, "..", "rvc"))
)
if not os.path.isdir(os.path.join(RVC_ROOT, "infer")):
    RVC_ROOT = SCRIPT_DIR  # 兜底：没找到 rvc 仓库就用脚本目录
os.chdir(RVC_ROOT)
if RVC_ROOT not in sys.path:
    sys.path.insert(0, RVC_ROOT)

os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(SCRIPT_DIR, "tmp"))
os.makedirs(TMP_ROOT, exist_ok=True)

API_PORT = int(os.environ.get("API_PORT", "8010"))
SEPARATE_MAX_SEC = int(os.environ.get("SEPARATE_MAX_SEC", "90"))  # 超过则自动分段转换

# 训练音色模型目录（与文字驱动项目同款约定）：训练中心（换声模式）的交付包
# 交付模型\rvc\<角色名>\ 整个文件夹复制到这里即自动识别，无需注册/重启
TRAINED_DIR = os.environ.get("RVC_TRAINED_DIR", os.path.join(SCRIPT_DIR, "models"))

# ---------------- 基础库 + RVC 仓库模块（必须已在 RVC_ROOT 下） ----------------
import librosa
import numpy as np
import soundfile as sf

from configs.config import Config
from infer.vc.modules import VC
from infer.vc.utils import get_index_path_from_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rvc_character")

CHARACTER_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fff]+$")

# 角色目标音高（Hz）：自动音高匹配时按“素材实际音高 → 该值”计算变调半音数
CHAR_TARGET_PITCH_HZ = {
    "Nahida": 220, "NahidaCN": 220, "Paimon": 225, "testnv": 205,
    "dabing": 130, "liejun": 130, "songwukong": 150,
}

_rvc_lock = threading.Lock()
# 当前驻留显存的模型（绝对路径）。同一模型连续转换不重复加载；
# 换模型时先卸载旧的再加载新的——显存里同时只有一个音色模型。
_current_model = None

# 常驻 RVC 推理实例（模型在 get_vc 时按角色懒加载）
config = Config()
vc = VC(config)


def _model_unload_locked():
    """在 _rvc_lock 内调用：把当前音色模型移出显存。

    卸载走 RVC 官方 vc.get_vc("")（内部清推理图缓存 + hubert + 权重并
    empty_cache），但它会把 net_g/cpt/n_spk/hubert_model/tgt_sr 属性
    **delattr** 掉，而 get_vc 用 `if self.net_g is not None` 判断，属性
    缺失会让下次加载直接 AttributeError——所以卸载后立刻把这些属性
    还原成 None。不要手工拆 net_g/pipeline：部分拆除后再次装载会在
    rmvpe 加载时段错误（实测 0xC0000005）。
    """
    global _current_model
    if _current_model is None:
        return
    vc.get_vc("")  # RVC 官方卸载分支
    for attr in ("net_g", "cpt", "n_spk", "hubert_model", "tgt_sr"):
        setattr(vc, attr, None)
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _current_model = None
    logger.info("音色模型已从显存卸载")


def _model_load(path_abs):
    """在 _rvc_lock 内调用：保证显存里只驻留这一个模型（换模型先卸载旧的）。"""
    global _current_model
    if _current_model == path_abs:
        return  # 已是当前模型，直接复用（多段连续转换不重载）
    _model_unload_locked()
    vc.get_vc(os.path.basename(path_abs))
    _current_model = path_abs
    logger.info("已加载音色模型：%s", os.path.basename(path_abs))


def discover_trained():
    """扫描训练音色模型目录（rvc_service\\models\\<角色名>\\），返回 {角色名: {pth, index}}。

    约定模仿文字驱动项目 tts_api：每角色一个子目录，优先 <角色名>.pth /
    <角色名>.index，找不到时用目录内任意 .pth / .index 兜底；索引可选。
    每次请求现扫现用（目录扫描开销极小），复制交付包进来"刷新即出现"。
    """
    out = {}
    if not os.path.isdir(TRAINED_DIR):
        return out
    for name in sorted(os.listdir(TRAINED_DIR)):
        d = os.path.join(TRAINED_DIR, name)
        if not (os.path.isdir(d) and CHARACTER_RE.match(name or "")):
            continue
        files = os.listdir(d)
        pth = os.path.join(d, name + ".pth")
        if not os.path.isfile(pth):
            pths = sorted(x for x in files if x.lower().endswith(".pth"))
            pth = os.path.join(d, pths[0]) if pths else ""
        if not pth:
            continue  # 没有 .pth 的目录不算角色
        idx = os.path.join(d, name + ".index")
        if not os.path.isfile(idx):
            idxs = sorted(x for x in files if x.lower().endswith(".index"))
            idx = os.path.join(d, idxs[0]) if idxs else ""
        out[name] = {"pth": pth, "index": idx or None}
    return out


def normalize_wav(src, workdir):
    """统一转成 44.1kHz 单声道 wav（RVC 内部会再按模型采样率处理）。"""
    dst = os.path.join(workdir, "normalized.wav")
    y, _ = librosa.load(src, sr=44100, mono=True)
    sf.write(dst, y, 44100)
    return dst


def measure_median_pitch(wav_path):
    """用 parselmouth 测人声中位数音高（Hz）；帧太少返回 None。"""
    try:
        import parselmouth

        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        snd = parselmouth.Sound(y, sr)
        pitch = snd.to_pitch_ac(
            time_step=0.01, voicing_threshold=0.35,
            pitch_floor=50, pitch_ceiling=1100,
        )
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        if len(voiced) < 20:
            return None
        return float(np.median(voiced))
    except Exception:  # noqa: BLE001
        return None


def auto_f0_up_key(clean_wav, character, fallback):
    """按角色目标音高自动算变调半音数；测不到就退回手动值。"""
    src = measure_median_pitch(clean_wav)
    if not src or src <= 40:
        return fallback
    target = CHAR_TARGET_PITCH_HZ.get(character)
    if not target:
        return fallback
    key = int(round(12.0 * math.log2(target / src)))
    logger.info("自动音高匹配: 素材中位音高=%.1fHz -> %s 目标=%.0fHz 变调=%+d",
                src, character, target, key)
    return key


def rvc_convert(
    input_wav,
    character,
    f0_up_key,
    index_rate,
    protect,
    f0_method,
    resample_sr,
    rms_mix_rate,
    speaker_id,
    model_path=None,
    index_path=None,
):
    if not CHARACTER_RE.match(character or ""):
        raise ValueError("character 名称不合法（仅允许字母/数字/下划线/中文）")
    model_name = character + ".pth"
    if model_path is None:
        # 兜底角色：rvc\assets\weights\<角色名>.pth
        model_path = os.path.join(os.environ["weight_root"], model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "模型不存在: %s（请放入 %s）" % (model_path, os.path.abspath(os.environ["weight_root"]))
            )
    if index_path is None:
        # 自动找 index：先精确匹配 <角色名>.index（防止 Nahida/NahidaV2 这类前缀名互相串索引），
        # 找不到再退回 RVC 的模糊匹配
        simple = os.path.join(os.environ["outside_index_root"], character + ".index")
        if os.path.isfile(simple):
            index_path = simple
        else:
            index_path = get_index_path_from_model(model_name, int(speaker_id))

    with _rvc_lock:
        # get_vc 按 weight_root/<sid> 找模型；训练音色在 rvc_service\models 下，
        # 请求期间临时指过去（转换被 _rvc_lock 串行化，env 翻转无竞态）
        prev_weight_root = os.environ.get("weight_root")
        mp_abs = os.path.abspath(model_path)
        os.environ["weight_root"] = os.path.dirname(mp_abs)
        try:
            _model_load(mp_abs)
            if int(resample_sr) == 0:  # 0 = 自动使用模型原始采样率（40k/48k 模型都通用）
                resample_sr = int(vc.tgt_sr)
            info, opt = vc.vc_single(
                sid=int(speaker_id),
                input_audio_path=os.path.abspath(input_wav),
                f0_up_key=int(f0_up_key),
                f0_method=f0_method,
                file_index=index_path,
                index_rate=float(index_rate),
                resample_sr=int(resample_sr),
                rms_mix_rate=float(rms_mix_rate),
                protect=float(protect),
            )
        finally:
            if prev_weight_root is None:
                os.environ.pop("weight_root", None)
            else:
                os.environ["weight_root"] = prev_weight_root
        if not opt or opt[0] is None or opt[1] is None:
            raise RuntimeError("RVC 推理失败: %s" % info)
        tgt_sr, audio_np = opt  # 新版返回 numpy 音频 + 目标采样率
        out_wav = os.path.join(TMP_ROOT, "rvc_%s.wav" % uuid.uuid4().hex)
        sf.write(out_wav, audio_np, int(tgt_sr))
    return out_wav


def rvc_convert_long(
    input_wav, workdir, character, f0_up_key, index_rate,
    protect, f0_method, resample_sr, rms_mix_rate, speaker_id,
    model_path=None, index_path=None,
):
    """超长音频自动分段转换后拼接（电影/长录音用）。"""
    y, sr = librosa.load(input_wav, sr=44100, mono=True)
    chunk_len = SEPARATE_MAX_SEC * sr
    parts = []
    seg_sr = sr
    for start in range(0, len(y), chunk_len):
        seg = y[start:start + chunk_len]
        seg_path = os.path.join(workdir, "seg_%d.wav" % (start // chunk_len))
        sf.write(seg_path, seg, sr)
        out_seg = rvc_convert(
            seg_path, character, f0_up_key, index_rate, protect,
            f0_method, resample_sr, rms_mix_rate, speaker_id,
            model_path=model_path, index_path=index_path,
        )
        a, seg_sr = sf.read(out_seg)
        parts.append(a)
    final = np.concatenate(parts)
    out_wav = os.path.join(TMP_ROOT, "rvc_long_%s.wav" % uuid.uuid4().hex)
    sf.write(out_wav, final, int(seg_sr))
    return out_wav


# ---------------- FastAPI 服务 ----------------
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>功能A：RVC 二次元角色换声</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:720px;margin:28px auto;padding:0 16px;color:#222}
h1{font-size:22px;border-bottom:2px solid #2e86de;padding-bottom:8px}
.card{background:#f7f9fc;border:1px solid #e0e3e8;border-radius:10px;padding:16px 18px;margin:14px 0}
label{display:block;margin:10px 0 4px;font-weight:600;font-size:14px}
input[type=file]{width:100%}
select{padding:6px;width:280px;font-size:14px}
button{background:#2e86de;color:#fff;border:none;padding:9px 26px;border-radius:6px;font-size:15px;cursor:pointer;margin-top:12px}
button:disabled{background:#aab}
.msg{font-size:13px;margin-top:8px;min-height:18px}
.err{color:#c0392b}.ok{color:#1e8e3e}
.tip{background:#fff7e6;border:1px solid #f0d48a;border-radius:8px;padding:8px 12px;font-size:12.5px;color:#6b5310;margin-top:12px}
a{color:#2e86de}</style>
</head>
<body>
<h1>🎮 功能A：RVC 二次元角色换声</h1>
<div class="card">
  <label>素材音频（wav/mp3/m4a）</label>
  <input type="file" id="audio" accept="audio/*">
  <label><input type="checkbox" id="auto_pitch" checked> 自动匹配音高（推荐，按素材实际音高对准角色）</label>
  <label>角色</label>
  <select id="character"></select>
  <button id="btn">开始转换</button>
  <div class="msg" id="msg"></div>
  <div id="result" style="display:none">
    <p><b>转换结果：</b></p>
    <audio id="player" controls style="width:100%"></audio>
    <br><a id="download" download="converted.wav">⬇ 下载 wav</a>
  </div>
  <div class="tip">带背景音乐/伴奏的素材请先在「换声工作台」（端口 8000）的批量换声里处理（会自动人声分离）。</div>
</div>
<script>
const CN = {
  "Nahida": "纳西妲·中配", "NahidaCN": "纳西妲·中配2", "Paimon": "派蒙·中配",
  "dabing": "大彬", "liejun": "列军", "songwukong": "孙悟空", "testnv": "测试女声"
};
async function loadModels(){
  const r = await fetch('/models'); const j = await r.json();
  document.getElementById('character').innerHTML = j.models.map(m =>
    '<option value="' + m + '">' + (CN[m] || m) + '</option>').join('');
}
loadModels();
document.getElementById('btn').addEventListener('click', async () => {
  const btn = document.getElementById('btn'), msg = document.getElementById('msg');
  if (!document.getElementById('audio').files[0]){ msg.className='err'; msg.textContent='请先选择素材音频'; return; }
  btn.disabled = true; msg.className=''; msg.textContent='转换中，请稍候…（首次加载模型较慢）';
  const fd = new FormData();
  fd.append('audio', document.getElementById('audio').files[0]);
  fd.append('character', document.getElementById('character').value);
  if (document.getElementById('auto_pitch').checked) fd.append('auto_pitch', 'true');
  try {
    const resp = await fetch('/convert', {method:'POST', body: fd});
    if (!resp.ok){ const t = await resp.text(); msg.className='err'; msg.textContent='失败：' + t.slice(0,200); return; }
    const blob = await resp.blob(); const url = URL.createObjectURL(blob);
    document.getElementById('player').src = url;
    document.getElementById('download').href = url;
    document.getElementById('result').style.display = 'block';
    msg.className='ok'; msg.textContent='完成 ✓';
  } catch(e){ msg.className='err'; msg.textContent='请求失败：' + e; }
  finally { btn.disabled = false; }
});
</script>
</body>
</html>"""

app = FastAPI(title="RVC Character Voice API")


def cleanup(workdir):
    shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "rvc-character",
        "device": config.device,
        "dtype": str(config.dtype),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/models")
def list_models():
    root = os.path.join(RVC_ROOT, os.environ["weight_root"])
    names = sorted(n[:-4] for n in os.listdir(root) if n.endswith(".pth"))
    return {"models": names, "trained": sorted(discover_trained().keys())}


@app.get("/model/status")
def model_status():
    import torch

    reserved = (
        torch.cuda.memory_reserved(0) // (1024 * 1024)
        if torch.cuda.is_available() else 0
    )
    return {
        "loaded": os.path.basename(_current_model) if _current_model else None,
        "vram_reserved_mb": reserved,
    }


@app.post("/model/unload")
def model_unload():
    """把当前音色模型从显存卸载（换声任务完成后由工作台调用，释放显卡）。"""
    with _rvc_lock:
        _model_unload_locked()
        import torch

        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
        else:
            reserved = 0
    return {"unloaded": True, "vram_reserved_mb": reserved}


@app.post("/convert")
def convert(
    audio: UploadFile = File(...),
    character: str = Form(...),
    f0_up_key: int = Form(0),
    auto_pitch: bool = Form(True),
    f0_method: str = Form("rmvpe"),
    index_rate: float = Form(0.75),
    protect: float = Form(0.33),
    resample_sr: int = Form(0),  # 0 = 自动匹配模型采样率（推荐，不用记 40k/48k）
    rms_mix_rate: float = Form(0.25),
    speaker_id: int = Form(0),
    background: BackgroundTasks = None,
):
    workdir = tempfile.mkdtemp(prefix="rvc_", dir=TMP_ROOT)
    try:
        raw_ext = os.path.splitext(audio.filename or "audio.wav")[1].lower()
        raw_file = os.path.join(workdir, "input" + raw_ext)
        with open(raw_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        clean_file = normalize_wav(raw_file, workdir)

        try:
            dur = sf.info(clean_file).frames / sf.info(clean_file).samplerate
        except Exception:  # noqa: BLE001
            dur = 0

        tinfo = discover_trained().get(character)
        if tinfo:
            # 训练音色（训练中心-换声模式交付的真人模型）：
            # 只换音色，音高/语调/时长保持原样——强制 f0_up_key=0，忽略 auto_pitch
            logger.info("训练音色 %s：f0_up_key=0（忽略 auto_pitch），只换音色", character)
            convert_kwargs = dict(model_path=tinfo["pth"], index_path=tinfo["index"])
            f0_key = 0
        else:
            # 自动音高匹配：按素材实际音高对准角色音色，比固定变调更像（二次元角色用）
            f0_key = auto_f0_up_key(clean_file, character, int(f0_up_key))
            convert_kwargs = {}

        if dur > SEPARATE_MAX_SEC:
            out_wav = rvc_convert_long(
                clean_file, workdir, character, f0_key, index_rate, protect,
                f0_method, resample_sr, rms_mix_rate, speaker_id,
                **convert_kwargs
            )
        else:
            out_wav = rvc_convert(
                clean_file, character, f0_key, index_rate, protect,
                f0_method, resample_sr, rms_mix_rate, speaker_id,
                **convert_kwargs
            )

        if background is not None:
            background.add_task(cleanup, workdir)
        return FileResponse(out_wav, media_type="audio/wav", filename="converted.wav")
    except HTTPException:
        cleanup(workdir)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("转换失败")
        cleanup(workdir)
        raise HTTPException(500, "转换失败: %s" % exc)


if __name__ == "__main__":
    import uvicorn

    # 必须单 worker：模型常驻显存
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, workers=1)