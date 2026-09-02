# -*- coding: utf-8 -*-
"""
统一换声控制台（端口 8000）
==========================
页面功能：
  1. 批量换声：给一个文件夹 → 分析每个音频/视频里有几个说话人 →
     为每个说话人分配角色 → 自动逐段换声 → 输出 wav / 混流后的视频
  2. 单文件换声：上传一个文件 + 选角色 → 换声
  3. 输出文件浏览：ziliao\\输出 下的最近结果

依赖引擎服务（换声动作转发给它们，本服务只做编排 + 说话人检测）：
  A RVC  8010   C SoVITS 8030   D GPT-SoVITS 8040   B OpenVoice 8020

运行（runtime\\py312）：
    runtime\\py312\\python.exe hub\\server.py
"""

import hashlib
import logging
import os
import shutil
import sys
import threading
import time
import uuid

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# 以脚本方式运行（python hub/server.py）时，嵌入版 Python 不会自动把脚本目录
# 加入 sys.path，这里显式补上，保证 diarize/pipeline/roles 可导入。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diarize  # noqa: E402,F401
import pipeline  # noqa: E402,F401
import roles  # noqa: E402,F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HUB_DIR, "web")
PREVIEW_DIR = os.path.join(HUB_DIR, "tmp", "previews")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ziliao", "输出")
API_PORT = int(os.environ.get("HUB_PORT", "8000"))

AUDIO_EXTS = pipeline.AUDIO_EXTS
VIDEO_EXTS = pipeline.VIDEO_EXTS
ALL_EXTS = AUDIO_EXTS + VIDEO_EXTS

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("hub")

app = FastAPI(title="换声工作台")

# ---------------- 任务状态（一次一个批量任务） ----------------
_state = {
    "running": False, "task": "", "step": "", "ok": False, "error": "",
    "log": [], "results": [], "total": 0, "done": 0,
}
_lock = threading.Lock()


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _lock:
        _state["log"].append(line)
        _state["log"] = _state["log"][-800:]
    logger.info(msg)


def set_state(**kw):
    with _lock:
        _state.update(kw)


def start_task():
    """原子地开始任务：已运行则拒绝。"""
    with _lock:
        if _state["running"]:
            return False
        _state.update(running=True, ok=False, error="", log=[], results=[],
                      task="", step="", total=0, done=0)
    return True


# ---------------- 工具 ----------------
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _preview_wav(src, sr=44100):
    """提取/转码为预览 wav（hub/tmp/previews），返回文件名。"""
    key = _md5(src) + "_" + str(os.path.getsize(src))
    name = key + ".wav"
    dst = os.path.join(PREVIEW_DIR, name)
    if not os.path.isfile(dst):
        pipeline.extract_audio(src, dst)
    return name


def _list_outputs(limit=60):
    """列出 ziliao\\输出 下最近的结果文件。"""
    out = []
    if not os.path.isdir(OUTPUT_ROOT):
        return out
    for root, _, names in os.walk(OUTPUT_ROOT):
        for n in sorted(names):
            if n.lower().endswith((".wav", ".mp4", ".m4a", ".flac")):
                p = os.path.join(root, n)
                rel = os.path.relpath(p, OUTPUT_ROOT)
                try:
                    dur = pipeline.duration_sec(p)
                except Exception:  # noqa: BLE001
                    dur = 0
                out.append({
                    "name": n, "rel": rel.replace("\\", "/"),
                    "size": os.path.getsize(p), "dur": round(dur, 1),
                    "mtime": os.path.getmtime(p),
                })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out[:limit]


def _safe_output_path(rel):
    """把相对路径安全地映射到 OUTPUT_ROOT 下。"""
    p = os.path.normpath(os.path.join(OUTPUT_ROOT, rel.replace("/", os.sep)))
    if not p.startswith(OUTPUT_ROOT):
        raise ValueError("非法路径")
    return p


# ---------------- 页面 ----------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>换声工作台</h1><p>缺少 web/index.html</p>")


# ---------------- API：状态 / 扫描 / 分析 / 批量 / 单文件 ----------------
@app.get("/api/status")
def status():
    svc = {}
    for eng in ("A", "B", "C", "D"):
        port = roles.PORTS[eng]
        svc[eng] = {
            "name": {"A": "RVC 二次元角色", "B": "OpenVoice 克隆",
                     "C": "SoVITS 中配", "D": "GPT-SoVITS 中配"}[eng],
            "port": port, "online": roles._port_open(port),
        }
    return {
        "services": svc,
        "roles": roles.load_roles(),
        "outputs": _list_outputs(),
        "task": {k: _state[k] for k in ("running", "step", "ok", "error", "results",
                                        "total", "done")},
    }


@app.post("/api/scan")
def scan(req: dict):
    try:
        files = pipeline.collect_folder(req.get("dir", ""))
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    items = []
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        items.append({
            "path": p, "name": os.path.basename(p),
            "ext": ext, "size": os.path.getsize(p),
            "is_video": ext in VIDEO_EXTS,
            "dur": round(pipeline.duration_sec(p), 1),
        })
    return {"dir": req.get("dir", ""), "files": items}


@app.post("/api/analyze")
def analyze(req: dict):
    """对指定文件逐个做说话人检测。返回每个文件的分段信息 + 预览音频。"""
    files = req.get("files") or []
    if not files:
        return JSONResponse({"detail": "没有要分析的文件"}, status_code=400)
    out = []
    for i, f in enumerate(files):
        path = f.get("path", "")
        if not os.path.isfile(path):
            out.append({"path": path, "name": os.path.basename(path), "error": "文件不存在"})
            continue
        try:
            preview = _preview_wav(path)
            y, sr = sf.read(os.path.join(PREVIEW_DIR, preview), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            d = diarize.diarize(np.asarray(y, dtype="float32"), sr)
            d["path"] = path
            d["name"] = os.path.basename(path)
            d["preview"] = preview
            d["is_video"] = os.path.splitext(path)[1].lower() in VIDEO_EXTS
            d["dur"] = round(len(y) / sr, 1)
            out.append(d)
        except Exception as exc:  # noqa: BLE001
            logger.exception("分析失败: %s", path)
            out.append({"path": path, "name": os.path.basename(path), "error": str(exc)})
    return {"results": out}


@app.get("/api/preview/{name}")
def preview(name: str):
    p = os.path.join(PREVIEW_DIR, os.path.basename(name))
    if not os.path.isfile(p):
        return JSONResponse({"detail": "预览不存在"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


def run_batch_task(dirpath, items, opts):
    """后台线程：逐个文件处理。"""
    task_id = time.strftime("batch_%Y%m%d_%H%M%S")
    out_root = os.path.join(OUTPUT_ROOT, task_id)
    os.makedirs(out_root, exist_ok=True)
    set_state(task=task_id, total=len(items))
    results = []
    base = os.path.join(PROJECT_ROOT, "ziliao", "tmp_batch")
    os.makedirs(base, exist_ok=True)
    try:
        for i, item in enumerate(items, 1):
            path = item.get("path", "")
            speakers = item.get("speakers") or {}
            if not os.path.isfile(path):
                log("✘ 文件不存在：%s" % path)
                continue
            set_state(step="[%d/%d] %s" % (i, len(items), os.path.basename(path)))
            workdir = os.path.join(base, uuid.uuid4().hex[:12])
            os.makedirs(workdir, exist_ok=True)
            # 把角色 id 映射成 role dict
            role_map = {}
            all_roles = {r["id"]: r for r in roles.load_roles()}
            for label, rid in speakers.items():
                r = all_roles.get(rid)
                if r:
                    role_map[str(label)] = r
            try:
                separate = bool(opts.get("separate", True))
                out_file = pipeline.process_file(path, role_map, workdir, separate,
                                                 out_root, log)
                if out_file:
                    results.append({
                        "name": os.path.basename(out_file),
                        "rel": os.path.relpath(out_file, OUTPUT_ROOT).replace("\\", "/"),
                        "dur": round(pipeline.duration_sec(out_file), 1),
                        "size": os.path.getsize(out_file),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.exception("处理失败: %s", path)
                log("✘ %s 处理失败：%s" % (os.path.basename(path), exc))
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            set_state(done=i)
        set_state(running=False, ok=True, step="全部完成", results=results)
        log("完成：共输出 %d 个文件 → %s" % (len(results), out_root))
    except Exception as exc:  # noqa: BLE001
        logger.exception("批量任务失败")
        set_state(running=False, ok=False, error="任务失败：%s" % exc)
        log("✘ 任务失败：%s" % exc)


@app.post("/api/batch")
def batch(req: dict):
    dirpath = req.get("dir", "")
    items = req.get("files") or []
    opts = req.get("opts") or {}
    if not os.path.isdir(dirpath):
        return JSONResponse({"detail": "文件夹不存在：%s" % dirpath}, status_code=400)
    if not items:
        return JSONResponse({"detail": "没有选择要处理的文件"}, status_code=400)
    # 校验文件都在该文件夹下
    for it in items:
        p = it.get("path", "")
        if not p.startswith(os.path.normpath(dirpath)):
            return JSONResponse({"detail": "文件不在所选文件夹内：%s" % p}, status_code=400)
    if not start_task():
        return JSONResponse({"detail": "已有任务在处理中，请稍候"}, status_code=400)
    threading.Thread(target=run_batch_task, args=(dirpath, items, opts), daemon=True).start()
    return {"message": "任务已开始（%d 个文件）" % len(items)}


@app.get("/api/task")
def task():
    with _lock:
        return {k: _state[k] for k in ("running", "step", "ok", "error", "log",
                                       "results", "total", "done", "task")}


@app.get("/api/output/file")
def output_file(rel: str):
    try:
        p = _safe_output_path(rel)
    except ValueError:
        return JSONResponse({"detail": "非法路径"}, status_code=400)
    if not os.path.isfile(p):
        return JSONResponse({"detail": "文件不存在"}, status_code=404)
    ext = os.path.splitext(p)[1].lower()
    media = "audio/wav" if ext == ".wav" else "video/mp4" if ext == ".mp4" else "application/octet-stream"
    return FileResponse(p, media_type=media, filename=os.path.basename(p))


@app.post("/api/output/delete")
def output_delete(req: dict):
    """删除输出文件（或某个 batch_* 子目录）。rel 相对 ziliao\\输出。"""
    rel = (req.get("rel") or "").strip().lstrip("/\\")
    try:
        p = _safe_output_path(rel)
    except ValueError:
        return JSONResponse({"detail": "非法路径"}, status_code=400)
    if os.path.normpath(p) == os.path.normpath(OUTPUT_ROOT):
        return JSONResponse({"detail": "不能删除输出根目录"}, status_code=400)
    if not os.path.exists(p):
        return JSONResponse({"detail": "文件不存在"}, status_code=404)
    try:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)
    except OSError as exc:
        return JSONResponse({"detail": "删除失败：%s" % exc}, status_code=500)
    return {"ok": True, "deleted": rel}


@app.post("/api/convert_one")
async def convert_one(
    audio: UploadFile = File(...),
    role_id: str = Form(...),
    separate: bool = Form(False),
):
    """单文件换声：上传文件 + 角色 → 返回 wav（视频附带 mp4 下载链接）。"""
    r = roles.role_by_id(role_id)
    if r is None:
        return JSONResponse({"detail": "角色不存在：%s" % role_id}, status_code=400)
    if not roles._port_open(r["port"]):
        return JSONResponse({"detail": "引擎 %s（端口 %d）未启动，请先启动对应服务"
                                      % (r["engine_cn"], r["port"])}, status_code=400)
    workdir = os.path.join(PROJECT_ROOT, "ziliao", "tmp_one")
    os.makedirs(workdir, exist_ok=True)
    d = os.path.join(workdir, uuid.uuid4().hex[:12])
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, audio.filename or "input.wav")
    with open(src, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    is_video = os.path.splitext(src)[1].lower() in VIDEO_EXTS
    try:
        out_file = pipeline.process_file(
            src, {"0": r}, d, bool(separate), d, lambda m: logger.info(m))
        if out_file is None:
            return JSONResponse({"detail": "未检测到人声或没有可转换的段"}, status_code=422)
        rel = os.path.basename(out_file)
        return {"file": rel, "wav": "/api/one/download/%s/%s" % (os.path.basename(d), rel),
                "mp4": "/api/one/download/%s/%s" % (os.path.basename(d), os.path.basename(out_file))
                if is_video else None}
    except Exception as exc:  # noqa: BLE001
        logger.exception("单文件转换失败")
        return JSONResponse({"detail": "转换失败：%s" % exc}, status_code=500)


@app.get("/api/one/download/{task_dir}/{filename}")
def one_download(task_dir: str, filename: str):
    d = os.path.join(PROJECT_ROOT, "ziliao", "tmp_one", os.path.basename(task_dir))
    p = os.path.join(d, os.path.basename(filename))
    if not os.path.isfile(p):
        return JSONResponse({"detail": "文件不存在"}, status_code=404)
    ext = os.path.splitext(p)[1].lower()
    media = "video/mp4" if ext == ".mp4" else "audio/wav"
    return FileResponse(p, media_type=media, filename=os.path.basename(p))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "hub", "port": API_PORT}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, workers=1)
