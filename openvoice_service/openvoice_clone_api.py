# -*- coding: utf-8 -*-
"""
服务B：OpenVoice 任意人声克隆 API（独立部署）
=============================================
功能：上传素材音频 + 一段参考音频（5~30 秒清晰人声），
      输出音色变成参考人、内容保持素材音频的 wav。无需训练，零样本克隆。

引擎：OpenVoice V2（Python 3.10 + torch 2.1.2+cu118）
端口：默认 8020（环境变量 API_PORT 可改）

运行：
    runtime\\py310\\python.exe openvoice_service\\openvoice_clone_api.py

可选环境变量：
    OV_CKPT       checkpoints_v2 目录（默认：本文件同目录/checkpoints_v2）
    OV_PROCESSED  SE 缓存目录（默认：本文件同目录/processed）
    API_PORT      监听端口（默认 8020）
    OV_DEVICE     设备（默认 cuda:0；显存紧张可设 cpu）
    OV_WATERMARK  是否加水印（默认 1；0 可省显存/提速，但失去可检测性）
"""

import hashlib
import logging
import os
import shutil
import tempfile
import threading
import uuid  # noqa: F401

import librosa
import soundfile as sf
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("OV_CKPT", os.path.join(SCRIPT_DIR, "checkpoints_v2"))
PROCESSED_DIR = os.environ.get("OV_PROCESSED", os.path.join(SCRIPT_DIR, "processed"))
API_PORT = int(os.environ.get("API_PORT", "8020"))
DEVICE = os.environ.get("OV_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
ENABLE_WATERMARK = os.environ.get("OV_WATERMARK", "1") == "1"
TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(SCRIPT_DIR, "tmp"))

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("openvoice_clone_api")

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>功能B：OpenVoice 任意人声克隆</title>
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
<h1>🎙 功能B：OpenVoice 任意人声克隆</h1>
<div class="card">
  <label>素材音频（要改音色的内容，wav/mp3/m4a）</label>
  <input type="file" id="audio" accept="audio/*">
  <label>参考人声（5~30 秒清晰人声，单人、无音乐）</label>
  <input type="file" id="ref_audio" accept="audio/*">
  <label>相似度 tau（越小越像参考人，默认 0.15）：<span id="tau_v">0.15</span></label>
  <input type="range" id="tau" min="0.05" max="0.9" step="0.05" value="0.15"
         oninput="document.getElementById('tau_v').textContent=this.value">
  <br>
  <button id="btn">开始克隆</button>
  <div class="msg" id="msg"></div>
  <div id="result" style="display:none">
    <p><b>克隆结果：</b></p>
    <audio id="player" controls style="width:100%"></audio>
    <br><a id="download" download="cloned.wav">⬇ 下载 wav</a>
  </div>
</div>
<script>
document.getElementById('btn').addEventListener('click', async () => {
  const btn = document.getElementById('btn'), msg = document.getElementById('msg');
  if (!document.getElementById('audio').files[0] || !document.getElementById('ref_audio').files[0]){
    msg.className='err'; msg.textContent='请先选择素材音频和参考人声'; return;
  }
  btn.disabled = true; msg.className=''; msg.textContent='克隆中，请稍候…（首次加载模型较慢）';
  const fd = new FormData();
  fd.append('audio', document.getElementById('audio').files[0]);
  fd.append('ref_audio', document.getElementById('ref_audio').files[0]);
  fd.append('tau', document.getElementById('tau').value);
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

from openvoice import se_extractor  # noqa: E402
from openvoice.api import ToneColorConverter  # noqa: E402
from openvoice.mel_processing import spectrogram_torch  # noqa: E402

_converter = None
_convert_lock = threading.Lock()
_se_cache = {}
_se_lock = threading.Lock()


def get_converter():
    global _converter
    if _converter is None:
        config_path = os.path.join(CKPT, "converter", "config.json")
        ckpt_path = os.path.join(CKPT, "converter", "checkpoint.pth")
        if not os.path.isfile(config_path) or not os.path.isfile(ckpt_path):
            raise RuntimeError(
                "OpenVoice 模型缺失，请检查 %s（需要 converter/config.json 和 converter/checkpoint.pth）" % CKPT
            )
        # 官方 V2 用法：构造 ToneColorConverter 后 load_ckpt。
        # 水印模型 wavmark 需要联网下载；离线时自动降级为无水印（不影响功能）。
        try:
            _converter = ToneColorConverter(
                config_path, device=DEVICE, enable_watermark=ENABLE_WATERMARK
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("wavmark 水印模型不可用（%s），自动改为无水印模式", exc)
            _converter = ToneColorConverter(
                config_path, device=DEVICE, enable_watermark=False
            )
        _converter.load_ckpt(ckpt_path)
        logger.info("OpenVoice 模型加载完成 device=%s watermark=%s", DEVICE, ENABLE_WATERMARK)
    return _converter


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


def _target_se(ref_wav):
    """参考音频音色向量（进程内按文件哈希缓存，同一段参考只提一次）。"""
    key = _file_hash(ref_wav)
    with _se_lock:
        if key in _se_cache:
            return _se_cache[key]
    converter = get_converter()
    se = extract_se_robust(ref_wav, converter)
    with _se_lock:
        _se_cache[key] = se
    return se


def _speech_segments(y, sr, top_db=28, min_speech=0.6, max_chunk=12.0):
    """基于能量的 VAD：去掉首尾静音，把活动语音合并成 <=max_chunk 的段（秒级采样点）。"""
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    segs = []
    for s, e in intervals:
        dur = (e - s) / sr
        if dur < min_speech:
            continue
        s = s + int(0.08 * sr)
        e = e - int(0.05 * sr)
        if e <= s + int(min_speech * sr):
            continue
        segs.append((s, e))
    merged = []
    for s, e in segs:
        if merged and (e - merged[-1][0]) / sr <= max_chunk:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _se_from_samples(converter, y):
    """直接从采样点提取音色向量（不落盘，等价于 converter.extract_se）。"""
    device = converter.device
    y = torch.FloatTensor(y).unsqueeze(0).to(device)
    spec = spectrogram_torch(
        y,
        converter.hps.data.filter_length,
        converter.hps.data.sampling_rate,
        converter.hps.data.hop_length,
        converter.hps.data.win_length,
        center=False,
    )
    with torch.no_grad():
        g = converter.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1)
    return g


def extract_se_robust(wav_path, converter):
    """VAD 分段 + 多段平均的音色向量。整体语音不足一段时回退到整段提取。"""
    y, sr = librosa.load(wav_path, sr=converter.hps.data.sampling_rate, mono=True)
    segs = _speech_segments(y, sr)
    if not segs:
        logger.info("VAD 未找到足够语音段（%s），回退整段提取", os.path.basename(wav_path))
        return converter.extract_se([wav_path])
    vecs = [_se_from_samples(converter, y[s:e]) for s, e in segs]
    se = torch.stack(vecs).mean(0)
    logger.info("SE 提取 %s: %d 段语音", os.path.basename(wav_path), len(segs))
    return se


def normalize_wav(src, workdir, name):
    dst = os.path.join(workdir, name)
    y, _ = librosa.load(src, sr=44100, mono=True)
    sf.write(dst, y, 44100)
    return dst


app = FastAPI(title="OpenVoice Clone API")


def cleanup(workdir):
    shutil.rmtree(workdir, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "openvoice-clone", "device": DEVICE, "ckpt": CKPT}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/convert")
def convert(
    audio: UploadFile = File(...),
    ref_audio: UploadFile = File(...),
    tau: float = Form(0.15),
    twopass: bool = Form(False),
    message: str = Form("converted"),
    background: BackgroundTasks = None,
):
    workdir = tempfile.mkdtemp(prefix="ov_", dir=TMP_ROOT)
    try:
        raw_ext = os.path.splitext(audio.filename or "audio.wav")[1].lower()
        raw_file = os.path.join(workdir, "input" + raw_ext)
        with open(raw_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        ref_ext = os.path.splitext(ref_audio.filename or "ref.wav")[1].lower()
        ref_raw = os.path.join(workdir, "ref" + ref_ext)
        with open(ref_raw, "wb") as f:
            shutil.copyfileobj(ref_audio.file, f)

        input_wav = normalize_wav(raw_file, workdir, "input.wav")
        ref_wav = normalize_wav(ref_raw, workdir, "ref.wav")
        out_wav = os.path.join(workdir, "cloned.wav")

        converter = get_converter()
        with _convert_lock:
            # 源音频和目标音频都要提取音色向量（src_se 必填）；都走 VAD 分段提取
            source_se = extract_se_robust(input_wav, converter)
            target_se = _target_se(ref_wav)
            if twopass:
                # 两遍：第一遍输出作为新“源”再向目标转一遍，音色更贴参考，
                # 但会放大伪影（实测毛刺/机械感更明显），默认关闭。
                pass1_wav = os.path.join(workdir, "pass1.wav")
                converter.convert(
                    audio_src_path=input_wav,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=pass1_wav,
                    tau=tau,
                    message="",
                )
                pass1_se = extract_se_robust(pass1_wav, converter)
                converter.convert(
                    audio_src_path=pass1_wav,
                    src_se=pass1_se,
                    tgt_se=target_se,
                    output_path=out_wav,
                    tau=tau,
                    message=message,
                )
            else:
                # 默认单遍：更干净，伪影最少
                converter.convert(
                    audio_src_path=input_wav,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=out_wav,
                    tau=tau,
                    message=message,
                )

        if not os.path.isfile(out_wav):
            raise RuntimeError("OpenVoice 未生成输出文件")
        if background is not None:
            background.add_task(cleanup, workdir)
        return FileResponse(out_wav, media_type="audio/wav", filename="cloned.wav")
    except HTTPException:
        cleanup(workdir)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("转换失败")
        cleanup(workdir)
        raise HTTPException(500, "转换失败: %s" % exc)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, workers=1)
