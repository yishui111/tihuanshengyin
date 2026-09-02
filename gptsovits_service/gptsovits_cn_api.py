# -*- coding: utf-8 -*-
"""
服务D：GPT-SoVITS 中配角色换声 API（独立部署，默认端口 8040）
=============================================================
流程：素材音频 → 人声分离/去混响（可选）→ 语音识别(ASR)转成文字
      → GPT-SoVITS 用角色音色重新合成 → wav
这是目前中文二次元配音效果最好的管线（ASR+TTS 重合成）。

角色（一个角色一套 GPT+SoVITS 权重）：
    ayaka  神里绫华（中配）
    azhong Azhong（中配，角色待确认）
"""

import io
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid

import librosa
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GSV_ROOT = os.environ.get("GSV_ROOT", os.path.join(SCRIPT_DIR, "GPT-SoVITS"))
GSV_MODELS_DIR = os.environ.get("GSV_MODELS_DIR", os.path.join(SCRIPT_DIR, "models"))
GSV_ASR_DIR = os.environ.get("GSV_ASR_DIR", os.path.join(SCRIPT_DIR, "asr", "SenseVoiceSmall"))
API_PORT = int(os.environ.get("API_PORT", "8040"))
CHARACTER_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fff]+$")

TMP_ROOT = os.environ.get("TMP_ROOT", os.path.join(SCRIPT_DIR, "tmp"))
os.makedirs(TMP_ROOT, exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "tmp", "matplotlib"), exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(SCRIPT_DIR, "tmp", "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(SCRIPT_DIR, "tmp", "numba"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gptsovits_cn")

# ---------------- 角色注册表 ----------------
CHARACTERS = {
# == 训练中心自动注册（勿删） ==
    "songwukong": {
        "cn": "songwukong(训练中心)",
        "gpt": os.path.join(GSV_MODELS_DIR, "songwukong", "songwukong.ckpt"),
        "sovits": os.path.join(GSV_MODELS_DIR, "songwukong", "songwukong.pth"),
        "ref": os.path.join(GSV_MODELS_DIR, "songwukong", "ref.wav"),
        "ref_text": "飞仙极大闲来对夺奖金不是神仙怎么说出神仙的呢弟子车子木筏漂洋过海历尽千辛万苦方他来到",
    },

    "ayaka": {
        "cn": "神里绫华(中配·GPT-SoVITS)",
        "gpt": os.path.join(GSV_MODELS_DIR, "ayaka", "Ayaka-e15.ckpt"),
        "sovits": os.path.join(GSV_MODELS_DIR, "ayaka", "Ayaka_e8_s176.pth"),
        "ref": os.path.join(GSV_MODELS_DIR, "ayaka", "ref.wav"),
        "ref_text": "晚上好夜风舒畅会是一个良宵呢",
    },
    "azhong": {
        "cn": "Azhong(中配·GPT-SoVITS)",
        "gpt": os.path.join(GSV_MODELS_DIR, "azhong", "Azhong-e15.ckpt"),
        "sovits": os.path.join(GSV_MODELS_DIR, "azhong", "Azhong_e8_s320.pth"),
        "ref": os.path.join(GSV_MODELS_DIR, "azhong", "ref.wav"),
        "ref_text": "",
    },
}

# 启动前把 api.py 的参数填好（默认加载 ayaka），再导入 GPT-SoVITS api
os.chdir(GSV_ROOT)
if GSV_ROOT not in sys.path:
    sys.path.insert(0, GSV_ROOT)
sys.argv = [
    "api.py",
    "-s", CHARACTERS["ayaka"]["sovits"],
    "-g", CHARACTERS["ayaka"]["gpt"],
    "-dr", CHARACTERS["ayaka"]["ref"],
    "-dt", CHARACTERS["ayaka"]["ref_text"],
    "-dl", "zh",
    "-hb", os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base"),
    "-b", os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large"),
    "-p", str(API_PORT),
]
import api as gpt_api  # noqa: E402

_gsv_lock = threading.Lock()
_loaded_char = "ayaka"
_asr_model = None


def get_asr():
    global _asr_model
    if _asr_model is None:
        from funasr import AutoModel

        logger.info("加载语音识别 SenseVoice")
        _asr_model = AutoModel(model=GSV_ASR_DIR, trust_remote_code=False,
                               device="cuda", disable_update=True)
    return _asr_model


def transcribe(wav_path):
    res = get_asr().generate(input=wav_path)[0]
    text = res.get("text", "")
    # 去掉 SenseVoice 的 <|zh|> <|NEUTRAL|> 等标签
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()
    return text


def normalize_wav(src, workdir):
    dst = os.path.join(workdir, "normalized.wav")
    y, _ = librosa.load(src, sr=44100, mono=True)
    sf.write(dst, y, 44100)
    return dst


def gpt_sovits_tts(character, text, top_k, top_p, temperature, speed):
    """切换角色权重（需要时）并用 GPT-SoVITS 合成。返回 32000Hz numpy 音频。"""
    global _loaded_char
    info = CHARACTERS[character]
    ref_text = info["ref_text"] or open(
        os.path.join(os.path.dirname(info["ref"]), "ref_text.txt"), encoding="utf-8"
    ).read().strip()
    with _gsv_lock:
        if _loaded_char != character:
            logger.info("切换角色模型: %s", character)
            r = gpt_api.change_gpt_sovits_weights(
                gpt_path=info["gpt"], sovits_path=info["sovits"])
            if getattr(r, "status_code", 200) != 200:
                raise RuntimeError("切换模型失败: %s" % r.body)
            _loaded_char = character
        gen = gpt_api.get_tts_wav(
            ref_wav_path=info["ref"],
            prompt_text=ref_text,
            prompt_language="zh",
            text=text,
            text_language="zh",
            top_k=int(top_k), top_p=float(top_p),
            temperature=float(temperature), speed=float(speed),
            inp_refs=[], sample_steps=32, if_sr=False,
        )
        chunks = []
        sr = 32000
        for item in gen:
            data = item[0] if isinstance(item, tuple) else item
            if isinstance(data, bytes):
                a, s = sf.read(io.BytesIO(data), dtype="float32")
                sr = s
                chunks.append(a)
            elif hasattr(data, "cpu"):
                chunks.append(data.cpu().numpy())
            else:
                chunks.append(np.asarray(data))
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, sr


# ---------------- FastAPI 服务 ----------------
from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402

app = FastAPI(title="GPT-SoVITS 中配换声")

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>功能D：GPT-SoVITS 中配角色换声</title>
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
<h1>🎙 功能D：GPT-SoVITS 中配角色换声</h1>
<div class="card">
  <label>素材音频（wav/mp3/m4a）</label>
  <input type="file" id="audio" accept="audio/*">
  <label>角色</label>
  <select id="character"></select>
  <button id="btn">开始转换</button>
  <div class="msg" id="msg"></div>
  <div id="result" style="display:none">
    <p><b>转换结果：</b></p>
    <audio id="player" controls style="width:100%"></audio>
    <br><a id="download" download="converted.wav">⬇ 下载 wav</a>
  </div>
  <div class="tip">本功能先语音识别（ASR）再重新合成，音色最像角色，但句子会被"重新说一遍"；想保留原语气节奏请用功能C。</div>
</div>
<script>
const CN = {"ayaka":"神里绫华","azhong":"Azhong","songwukong":"孙悟空"};
async function loadModels(){
  const r = await fetch('/models'); const j = await r.json();
  document.getElementById('character').innerHTML = j.models.map(m =>
    '<option value="' + m + '">' + (CN[m] || m) + '</option>').join('');
}
loadModels();
document.getElementById('btn').addEventListener('click', async () => {
  const btn = document.getElementById('btn'), msg = document.getElementById('msg');
  if (!document.getElementById('audio').files[0]){ msg.className='err'; msg.textContent='请先选择素材音频'; return; }
  btn.disabled = true; msg.className=''; msg.textContent='转换中（识别+合成，约 30~90 秒），请稍候…';
  const fd = new FormData();
  fd.append('audio', document.getElementById('audio').files[0]);
  fd.append('character', document.getElementById('character').value);
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

    return {"status": "ok", "service": "gptsovits-cn",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu"}


@app.get("/models")
def list_models():
    return {"models": sorted(CHARACTERS.keys())}


@app.post("/convert")
def convert(
    audio: UploadFile = File(...),
    character: str = Form("ayaka"),
    speed: float = Form(1.0),
    top_k: int = Form(15),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
):
    if not CHARACTER_RE.match(character or ""):
        raise HTTPException(400, "character 名称不合法")
    if character not in CHARACTERS:
        raise HTTPException(404, "角色不存在: %s（可用角色见 GET /models）" % character)
    workdir = tempfile.mkdtemp(prefix="gpt_", dir=TMP_ROOT)
    try:
        raw_ext = os.path.splitext(audio.filename or "audio.wav")[1].lower()
        raw_file = os.path.join(workdir, "input" + raw_ext)
        with open(raw_file, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        clean_file = normalize_wav(raw_file, workdir)

        logger.info("语音识别中...")
        text = transcribe(clean_file)
        if not text:
            raise HTTPException(422, "语音识别没识别到内容，请确认素材是清晰的人声")
        if len(text) > 500:
            text = text[:500]
        logger.info("识别文本(%d字): %s", len(text), text[:60])

        audio_np, sr = gpt_sovits_tts(character, text, top_k, top_p, temperature, speed)
        if len(audio_np) == 0:
            raise HTTPException(500, "合成结果为空")
        out_wav = os.path.join(TMP_ROOT, "gpt_%s.wav" % uuid.uuid4().hex)
        sf.write(out_wav, audio_np, sr)
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