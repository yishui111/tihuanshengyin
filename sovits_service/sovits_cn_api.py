# -*- coding: utf-8 -*-
"""
服务C：SoVITS 中配角色换声 API（独立部署，默认端口 8030）
=========================================================
功能：上传素材音频 + 角色，输出换成中配角色音色的 wav。
      SoVITS 4.1 + 中文纳西妲模型（11.1 万步 + 特征检索），
      自然度比 RVC 高，男声转中文萝莉音更像。

运行：
    在仓库根目录运行：
    runtime\\py312\\python.exe sovits_service\\sovits_cn_api.py

模型目录（sovits_service\\models\\）：
    nahida41_G_111200.pth            SoVITS 4.1 中文纳西妲主模型
    nahida41.json                    配置（speech_encoder 已改为 vec768l12-onnx）
    nahida41_feature_and_index.pkl   特征检索索引

预训练资产（so-vits-svc-4.1-Stable\\pretrain\\）：
    vec-768-layer-12.onnx            contentvec 语音编码器（ONNX，免 fairseq）
    rmvpe.pt                         f0 音高提取
"""

import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid

import librosa
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOVITS_SRC = os.environ.get("SOVITS_SRC", os.path.join(SCRIPT_DIR, "so-vits-svc-4.1-Stable"))
MODELS_DIR = os.environ.get("SOVITS_MODELS_DIR", os.path.join(SCRIPT_DIR, "models"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", os.path.join(SCRIPT_DIR, "..", "runtime", "ffmpeg", "bin"))
if os.path.isdir(FFMPEG_BIN):
    os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

# onnxruntime-gpu 1.17 需要 CUDA 11.8 + cuDNN 8 的 dll（nvidia pip 包 + torch 自带）。
# 不注入这些目录会导致 onnx 回退 CPU，SoVITS 首次加载/转换极慢（像卡死）。
for _d in [
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "nvidia", "cudnn", "bin"),
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "nvidia", "cufft", "bin"),
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"),
    os.path.join(SCRIPT_DIR, "..", "runtime", "py312", "Lib", "site-packages", "torch", "lib"),
]:
    if os.path.isdir(_d):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
API_PORT = int(os.environ.get("API_PORT", "8030"))
SEPARATE_MAX_SEC = int(os.environ.get("SEPARATE_MAX_SEC", "90"))
CHARACTER_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fff]+$")

TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(SCRIPT_DIR, "tmp"))
os.makedirs(TMP_ROOT, exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "tmp", "matplotlib"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(SCRIPT_DIR, "tmp", "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(SCRIPT_DIR, "tmp", "numba"))
os.environ.setdefault("VERIFY_WORK", SCRIPT_DIR)

# SoVITS 用相对路径加载 pretrain 资产，必须把工作目录切到源码根
os.chdir(SOVITS_SRC)
if SOVITS_SRC not in sys.path:
    sys.path.insert(0, SOVITS_SRC)

from inference.infer_tool import Svc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sovits_cn")

# ---------------- 角色注册表（以后加角色就加一行） ----------------
# target_hz: 该角色典型说话音高（自动音高匹配用）
# feature_retrieval=True 用特征检索索引(.pkl)，False 用 kmeans(.pt)
CHARACTERS = {
    "nahida": {
        "cn": "纳西妲(中配·SoVITS4.1)",
        "model": os.path.join(MODELS_DIR, "nahida41_G_111200.pth"),
        "config": os.path.join(MODELS_DIR, "nahida41.json"),
        "index": os.path.join(MODELS_DIR, "nahida41_feature_and_index.pkl"),
        "spk": "nahida",
        "target_hz": 250,
        "feature_retrieval": True,
    },
    "klee": {
        "cn": "可莉(中配·SoVITS)",
        "model": os.path.join(MODELS_DIR, "klee_G.pth"),
        "config": os.path.join(MODELS_DIR, "klee.json"),
        "index": os.path.join(MODELS_DIR, "klee_kmeans.pt"),
        "spk": "klee",
        "target_hz": 280,
        "feature_retrieval": False,
    },
    "hutao": {
        "cn": "胡桃(中配·SoVITS)",
        "model": os.path.join(MODELS_DIR, "hutao_G.pth"),
        "config": os.path.join(MODELS_DIR, "hutao.json"),
        "index": os.path.join(MODELS_DIR, "hutao_kmeans.pt"),
        "spk": "hutao",
        "target_hz": 260,
        "feature_retrieval": False,
    },
    "yaoyao": {
        "cn": "瑶瑶(中配·SoVITS)",
        "model": os.path.join(MODELS_DIR, "yaoyao_G.pth"),
        "config": os.path.join(MODELS_DIR, "yaoyao.json"),
        "index": os.path.join(MODELS_DIR, "yaoyao_kmeans.pt"),
        "spk": "yaoyao",
        "target_hz": 250,
        "feature_retrieval": False,
    },
    "raiden": {
        "cn": "雷电将军(中配·SoVITS4.1)",
        "model": os.path.join(MODELS_DIR, "randenEi_G.pth"),
        "config": os.path.join(MODELS_DIR, "randenEi.json"),
        "index": os.path.join(MODELS_DIR, "randenEi_feature_and_index.pkl"),
        "spk": "randenEi",
        "target_hz": 190,
        "feature_retrieval": True,
    },
    "furina": {
        "cn": "芙宁娜(中配·SoVITS)",
        "model": os.path.join(MODELS_DIR, "furina_G.pth"),
        "config": os.path.join(MODELS_DIR, "furina.json"),
        "index": os.path.join(MODELS_DIR, "furina_kmeans.pt"),
        "spk": "fnn",
        "target_hz": 240,
        "feature_retrieval": False,
    },
}

_svc_cache = {}
_svc_lock = threading.Lock()


def get_svc(character):
    """按角色懒加载 SoVITS 模型；切换角色时释放上一个，省显存。"""
    with _svc_lock:
        if character in _svc_cache:
            return _svc_cache[character]
        import gc

        for k in list(_svc_cache):
            if k != character:
                logger.info("释放模型: %s", k)
                try:
                    _svc_cache[k].unload_model()
                except Exception:  # noqa: BLE001
                    pass
                del _svc_cache[k]
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        info = CHARACTERS[character]
        logger.info("加载 SoVITS 模型: %s (%s)", character, info["model"])
        svc = Svc(
            info["model"], info["config"], device="cuda",
            cluster_model_path=info["index"],
            feature_retrieval=info.get("feature_retrieval", False),
        )
        _svc_cache[character] = svc
        return svc


def normalize_wav(src, workdir):
    dst = os.path.join(workdir, "normalized.wav")
    y, _ = librosa.load(src, sr=44100, mono=True)
    sf.write(dst, y, 44100)
    return dst


def measure_median_pitch(wav_path):
    try:
        import parselmouth

        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        snd = parselmouth.Sound(y, sr)
        f0 = snd.to_pitch_ac(
            time_step=0.01, voicing_threshold=0.35,
            pitch_floor=50, pitch_ceiling=1100,
        ).selected_array["frequency"]
        voiced = f0[f0 > 0]
        if len(voiced) < 20:
            return None
        return float(np.median(voiced))
    except Exception:  # noqa: BLE001
        return None


def sovits_convert(clean_wav, character, tran, f0_method, cluster_infer_ratio,
                   noice_scale, auto_predict_f0):
    """切片推理（官方 slice_inference）：按静音切片、每段前后补 pad_seconds
    静音上下文再转换，最后裁掉补丁——解决直接整段推理时开头发虚/听不清的问题。"""
    svc = get_svc(character)
    spk = CHARACTERS[character]["spk"]
    with _svc_lock:
        audio = svc.slice_inference(
            raw_audio_path=os.path.abspath(clean_wav),
            spk=spk,
            tran=tran,
            slice_db=-35,
            cluster_infer_ratio=float(cluster_infer_ratio),
            auto_predict_f0=bool(auto_predict_f0),
            noice_scale=float(noice_scale),
            pad_seconds=1.0,
            f0_predictor=f0_method,
        )
        out_wav = os.path.join(TMP_ROOT, "sovits_%s.wav" % uuid.uuid4().hex)
        sf.write(out_wav, audio, svc.target_sample)
    return out_wav


# ---------------- FastAPI 服务 ----------------
from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

app = FastAPI(title="SoVITS 中配换声")

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>功能C：SoVITS 中配角色换声</title>
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
<h1>🎧 功能C：SoVITS 中配角色换声</h1>
<div class="card">
  <label>素材音频（wav/mp3/m4a）</label>
  <input type="file" id="audio" accept="audio/*">
  <label><input type="checkbox" id="auto_pitch" checked> 自动匹配音高（推荐）</label>
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
const CN = {"nahida":"纳西妲","klee":"可莉","hutao":"胡桃","yaoyao":"瑶瑶","raiden":"雷电将军","furina":"芙宁娜"};
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


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    import torch

    return {
        "status": "ok",
        "service": "sovits-cn",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }


@app.get("/models")
def list_models():
    return {"models": sorted(CHARACTERS.keys())}


@app.post("/convert")
def convert(
    audio: UploadFile = File(...),
    character: str = Form("nahida"),
    f0_up_key: int = Form(0),
    auto_pitch: bool = Form(True),
    f0_method: str = Form("rmvpe"),
    cluster_infer_ratio: float = Form(0.75),
    noice_scale: float = Form(0.4),
    auto_predict_f0: bool = Form(False),
):
    if not CHARACTER_RE.match(character or ""):
        raise HTTPException(400, "character 名称不合法")
    if character not in CHARACTERS:
        raise HTTPException(404, "角色不存在: %s（可用角色见 GET /models）" % character)
    workdir = tempfile.mkdtemp(prefix="svc_", dir=TMP_ROOT)
    try:
        raw_ext = os.path.splitext(audio.filename or "audio.wav")[1].lower()
        raw_file = os.path.join(workdir, "input" + raw_ext)
        with open(raw_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        clean_file = normalize_wav(raw_file, workdir)

        tran = int(f0_up_key)
        if auto_pitch:
            src = measure_median_pitch(clean_file)
            target = CHARACTERS[character]["target_hz"]
            if src and src > 40:
                tran = int(round(12.0 * math.log2(target / src)))
                logger.info("自动音高匹配: 素材=%.1fHz -> %s 目标=%.0fHz 变调=%+d",
                            src, character, target, tran)

        out_wav = sovits_convert(clean_file, character, tran, f0_method,
                                 cluster_infer_ratio, noice_scale, auto_predict_f0)
        return FileResponse(out_wav, media_type="audio/wav", filename="converted.wav")
    except HTTPException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("转换失败")
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, "转换失败: %s" % exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=API_PORT, workers=1)