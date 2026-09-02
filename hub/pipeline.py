# -*- coding: utf-8 -*-
"""
批量换声管线：文件夹里的音频/视频 → 分析说话人 → 按说话人换角色声
==================================================================
单个文件流程：
    1. ffmpeg 提取/转码为 44.1k 单声道 wav
    2. （可选）pymss 人声分离：去掉背景音乐，音乐轨保留待合成
    3. VAD + ECAPA 声纹聚类 → 每段语音打说话人标签
    4. 每个语音段按"说话人→角色"映射调用 A(RVC)/C(SoVITS) 服务转换
    5. 按原时间轴拼接（保留停顿/静音；分离过的再叠回音乐轨）
    6. 视频文件用 ffmpeg 把新音轨混流回视频（画面不变）

引擎角色路由见 roles.py；本模块只负责编排。
"""

import os
import shutil
import subprocess
import threading

import numpy as np
import soundfile as sf

try:
    from . import diarize
except ImportError:  # 以脚本方式运行（python hub/server.py）时
    import diarize  # noqa: F401

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(PROJECT_ROOT, "runtime", "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE = os.path.join(PROJECT_ROOT, "runtime", "ffmpeg", "bin", "ffprobe.exe")
PYMSS_MODEL_DIR = os.path.join(PROJECT_ROOT, "rvc_service", "pymss_models")
VOCAL_MODEL = "bs_roformer_voc_hyperacev2"

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".flv", ".ts", ".webm", ".m4v")

SEPARATE_MAX_SEC = int(os.environ.get("SEPARATE_MAX_SEC", "90"))
MIN_CONVERT_SEC = 0.5  # 短于该时长的语音段保留原声（避免转换 0.x 秒抖动）
FADE_SEC = 0.045  # 段边界交叉淡化时长（防爆音 + 让相邻段过渡自然）

_sep = None
_sep_lock = threading.Lock()


def duration_sec(path):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.DEVNULL)
        return float(out.decode().strip())
    except Exception:  # noqa: BLE001
        return 0.0


def extract_audio(src, dst_wav):
    """任意音视频 → 44.1k 单声道 wav。"""
    subprocess.run(
        [FFMPEG, "-y", "-i", src, "-ar", "44100", "-ac", "1", dst_wav],
        check=True, capture_output=True)


def mux_video(video_src, audio_wav, out_mp4):
    """把新音轨混流回视频（视频流原样复制，不重编码）。"""
    subprocess.run(
        [FFMPEG, "-y", "-i", video_src, "-i", audio_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_mp4],
        check=True, capture_output=True)


def _device():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 6 * 1024 ** 3:
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_separator():
    """pymss 人声分离模型（离线，进程内单例）。"""
    global _sep
    if _sep is None:
        with _sep_lock:
            if _sep is None:
                from pymss import MSSeparator
                os.makedirs(PYMSS_MODEL_DIR, exist_ok=True)
                _sep = MSSeparator.from_model_name(
                    VOCAL_MODEL, model_dir=PYMSS_MODEL_DIR, download=False,
                    source="hf-mirror", device=_device(), output_format="wav",
                    inference_params={"normalize": True},
                )
    return _sep


def separate_vocals(y, sr):
    """人声/伴奏分离（超长分段处理）。返回 (vocals, instrumental)。"""
    sep = get_separator()
    chunk_len = SEPARATE_MAX_SEC * sr
    v_parts, i_parts = [], []
    for s in range(0, len(y), chunk_len):
        seg = y[s:s + chunk_len]
        stems = sep.separate(seg, pbar=False, stems=["vocals"])
        v = np.asarray(stems["vocals"], dtype="float32")
        if v.ndim > 1:
            v = v.mean(axis=1)
        if len(v) < len(seg):
            v = np.pad(v, (0, len(seg) - len(v)))
        i_parts.append(seg - v)
        v_parts.append(v)
    vocals = np.concatenate(v_parts) if len(v_parts) > 1 else v_parts[0]
    inst = np.concatenate(i_parts) if len(i_parts) > 1 else i_parts[0]
    if _device() == "cuda":
        import torch
        torch.cuda.empty_cache()
    return vocals, inst


def _convert_segment(role, seg_wav, out_wav):
    """调用引擎服务转换单个语音段。role 来自 roles.load_roles()。"""
    import requests

    url = "http://127.0.0.1:%d/convert" % role["port"]
    with open(seg_wav, "rb") as f:
        files = {"audio": ("seg.wav", f, "audio/wav")}
        if role["engine"] == "A":
            data = {"character": role["character"], "auto_pitch": "true"}
        elif role["engine"] == "C":
            data = {"character": role["character"], "auto_pitch": "true"}
        else:
            raise ValueError("引擎 %s 不支持批量逐段换声（请选 RVC 或 SoVITS 角色）" % role["engine_cn"])
        r = requests.post(url, files=files, data=data, timeout=600)
    if r.status_code != 200:
        raise RuntimeError("引擎转换失败(%s): %s" % (role["engine_cn"], r.text[:300]))
    with open(out_wav, "wb") as f:
        f.write(r.content)


def _fit_to(audio, n):
    """把转换结果对齐到目标长度 n（换声引擎输出时长与输入基本一致，
    少量偏差用裁剪/补零，偏差大时做轻微变速）。"""
    if len(audio) == n:
        return audio
    diff = abs(len(audio) - n)
    if diff > int(0.03 * n) and len(audio) > 0:
        import librosa
        audio = librosa.effects.time_stretch(audio, rate=len(audio) / n)
    if len(audio) > n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio)))


def _fade(x, sr):
    n = min(int(FADE_SEC * sr), len(x) // 4)
    if n <= 0:
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype="float32")
    out = x.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def merge_segments(segs, sr, gap_sec=1.5):
    """合并相邻、同说话人、间隔不超过 gap_sec 的语音段。

    作用：避免"一句话一个段、逐个独立转换"造成的句间断裂/音高跳变——
    把同一人的连续话语合并成一个大段一起转换，更自然连贯。
    """
    if not segs:
        return segs
    out = []
    for s in segs:
        if (out and s["label"] == out[-1]["label"]
                and (s["start"] - out[-1]["end"]) / sr <= gap_sec):
            out[-1]["end"] = s["end"]
            out[-1]["dur"] = round((s["end"] - s["start"]) / sr, 2)
        else:
            out.append(dict(s))
    return out


def process_file(src, mapping, workdir, separate, out_root, log):
    """处理单个文件。

    mapping: {说话人编号(str): role dict}（来自 roles.load_roles()）
    返回输出文件路径；无人声返回 None。

    换声粒度：
      - 单说话人：整段一次转换（引擎内部自动切片，最自然连贯）
      - 多说话人：先按说话人合并相邻段（间隔≤1.5s），每段尽量长
    """
    import librosa

    base = os.path.splitext(os.path.basename(src))[0]
    is_video = os.path.splitext(src)[1].lower() in VIDEO_EXTS
    log("① 提取音频：%s（%.1f 秒）" % (base, duration_sec(src)))
    raw_wav = os.path.join(workdir, "input.wav")
    extract_audio(src, raw_wav)
    y, sr = sf.read(raw_wav, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype="float32")

    music = None
    if separate:
        log("② 人声分离（去背景音乐）...")
        vocals, music = separate_vocals(y, sr)
        y = vocals

    log("③ 说话人检测...")
    d = diarize.diarize(y, sr)
    if d["n_speakers"] == 0:
        log("  未检测到人声，跳过该文件")
        return None
    log("  检测到 %d 个说话人、%d 个语音段" % (d["n_speakers"], len(d["segments"])))
    mapped = set(int(k) for k in mapping.keys())
    unmapped = sorted({s["label"] for s in d["segments"] if s["label"] not in mapped})
    if unmapped:
        log("  说话人%s 未分配角色，这些段落保留原声" % "、".join(str(u) for u in unmapped))

    # ---- 确定转换段（尽量整段/大段，避免逐句切碎） ----
    if d["n_speakers"] == 1:
        role0 = mapping.get("0")
        segs = [{"start": 0, "end": len(y), "label": 0, "dur": round(len(y) / sr, 2)}] \
            if role0 is not None else []
    else:
        segs = merge_segments(d["segments"], sr)
    if segs:
        log("  共 %d 段待转换" % len(segs))

    converted = np.zeros(len(y), dtype="float32")
    count = 0
    for seg in segs:
        role = mapping.get(str(seg["label"]))
        if role is None:
            continue  # 该说话人未分配角色 → 保留原声
        a, b = seg["start"], seg["end"]
        if b - a < int(MIN_CONVERT_SEC * sr):
            continue
        seg_wav = os.path.join(workdir, "seg.wav")
        sf.write(seg_wav, y[a:b], sr)
        out_wav = os.path.join(workdir, "seg_out.wav")
        log("  段 [%.1f-%.1f] 说话人%d → %s(%s)" % (
            a / sr, b / sr, seg["label"], role["name"], role["engine_cn"]))
        _convert_segment(role, seg_wav, out_wav)
        out_y, out_sr = sf.read(out_wav, dtype="float32")
        if out_y.ndim > 1:
            out_y = out_y.mean(axis=1)
        if out_sr != sr:
            out_y = librosa.resample(out_y, orig_sr=out_sr, target_sr=sr)
        fit = _fit_to(np.asarray(out_y, dtype="float32"), b - a)
        fit = _fade(fit, sr)
        converted[a:b] = fit
        count += 1
    if count == 0:
        log("  没有可转换的语音段（说话人未分配角色或段过短）")
        return None

    if music is not None:
        final = music.astype("float32") + converted
    else:
        final = y.copy().astype("float32")
        for seg in segs:
            role = mapping.get(str(seg["label"]))
            a, b = seg["start"], seg["end"]
            if role is not None and b - a >= int(MIN_CONVERT_SEC * sr):
                final[a:b] = converted[a:b]
    peak = float(np.abs(final).max())
    if peak > 0.99:
        final = final * (0.95 / peak)

    tag = "_".join(role["name"] for role in sorted(
        mapping.values(), key=lambda r: r["name"]))
    out_wav = os.path.join(out_root, "%s_%s.wav" % (base, tag))
    sf.write(out_wav, final, sr)
    log("④ 已生成：%s" % os.path.basename(out_wav))

    if is_video:
        out_mp4 = os.path.join(out_root, "%s_%s.mp4" % (base, tag))
        log("⑤ 混流回视频：%s" % os.path.basename(out_mp4))
        mux_video(src, out_wav, out_mp4)
        return out_mp4
    return out_wav


def collect_folder(dirpath):
    """递归收集文件夹内所有音频/视频文件。"""
    if not os.path.isdir(dirpath):
        raise ValueError("文件夹不存在：%s" % dirpath)
    files = []
    for root, _, names in os.walk(dirpath):
        for n in sorted(names):
            if n.lower().endswith(AUDIO_EXTS + VIDEO_EXTS):
                files.append(os.path.join(root, n))
    if not files:
        raise ValueError("该文件夹里没有音频/视频文件")
    return files
