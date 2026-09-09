# -*- coding: utf-8 -*-
"""
说话人检测（说话人分离）：pyannote 3.1 管线，失败时回退 VAD+ECAPA 聚类
=======================================================================
主后端：pyannote speaker-diarization-3.1（神经网络 VAD + 子段声纹 + 聚类，
段内换人/轻声/带噪场景明显更准），模型离线缓存在
    runtime\\cache\\pyannote\\segmentation-3.0\\
    runtime\\cache\\pyannote\\wespeaker-voxceleb-resnet34-LM\\
    runtime\\cache\\pyannote\\speaker-diarization-3.1\\config.yaml
模型文件缺失或推理出错时自动回退旧管线：
    能量 VAD 分段 → ECAPA 声纹（speechbrain，缓存 runtime\\cache\\hf_speaker_model）
    → 余弦距离层次聚类（手工阈值规则，保守易漏分，保留作兜底）

依赖（runtime\\py312 已内置）：
    librosa / numpy / torch / speechbrain(旧后端) / pyannote.audio>=3.1(新后端)
"""

import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "runtime", "cache", "huggingface")
SPK_SAVEDIR = os.path.join(PROJECT_ROOT, "runtime", "cache", "hf_speaker_model")
PYANNOTE_CONFIG = os.path.join(
    PROJECT_ROOT, "runtime", "cache", "pyannote",
    "speaker-diarization-3.1", "config.yaml")

# 离线优先：pyannote 模型已全部本地化，禁止运行时联网下载
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# pyannote 不走 HF_HOME，它用自己的 PYANNOTE_CACHE（默认 ~/.cache/torch/pyannote，
# 见 pyannote/audio/core/model.py 的 CACHE_DIR）；必须在其被导入前指到 runtime 内
os.environ.setdefault(
    "PYANNOTE_CACHE",
    os.path.join(PROJECT_ROOT, "runtime", "cache", "pyannote", "hf"))

# 模型进程内只加载一次（懒加载）
_model = None
_device = None
_pn_pipeline = None

# 关键：torch/pyannote 必须在本进程里先于 librosa 导入——librosa 的依赖链
# 会先加载自己的 CUDA 相关 DLL，之后 torch 的 caffe2_nvrtc.dll 因依赖被占
# 解析失败（WinError 126）。hub/server.py 的导入顺序保证 diarize 早于
# pipeline(librosa)，这里在模块导入时即完成预热；失败只记下，不阻塞导入。
try:
    import time as _time

    _torch = None
    _PyannotePipeline = None
    _PN_IMPORT_ERR = None
    for _attempt in range(3):
        try:
            import torch as _torch_mod  # noqa: F401
            from pyannote.audio import Pipeline as _pn_mod  # noqa: F401

            # pyannote 3.3.2 的 checkpoint 含 TorchVersion 等 pickle 全局，
            # torch>=2.6 默认 weights_only=True 会拒绝加载（lightning 还会
            # 显式传该参数）。模型文件全部本地缓存且 sha256 校验过，
            # 这里强制恢复旧行为。
            _orig_torch_load = _torch_mod.load

            def _torch_load_compat(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            _torch_mod.load = _torch_load_compat
            _torch = _torch_mod
            _PyannotePipeline = _pn_mod
            break
        except OSError as _e:
            # Windows 上 torch 的 CUDA DLL 偶发被杀软/其它进程瞬时占用，
            # 等 1 秒重试；其余异常不重试
            _PN_IMPORT_ERR = _e
            if "dll" not in str(_e).lower():
                break
            _time.sleep(1.0)
except Exception as _e:  # noqa: BLE001
    _torch = None
    _PyannotePipeline = None
    _PN_IMPORT_ERR = _e


def get_device():
    global _device
    if _device is None:
        import torch

        _device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    return _device


def get_model():
    """ECAPA-TDNN 声纹模型（旧后端兜底用，离线加载，进程内单例）。"""
    global _model
    if _model is None:
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy

        _model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SPK_SAVEDIR,
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": get_device()},
        )
    return _model


def get_pyannote_pipeline():
    """pyannote speaker-diarization-3.1 管线（进程内单例）。

    模型未下载（config.yaml 不存在）或 pyannote 导入失败时返回 None，
    调用方回退旧后端。
    """
    global _pn_pipeline
    if _PyannotePipeline is None:
        return None
    if _pn_pipeline is None and not os.path.exists(PYANNOTE_CONFIG):
        return None
    if _pn_pipeline is None:
        _pn_pipeline = _PyannotePipeline.from_pretrained(PYANNOTE_CONFIG)
        _pn_pipeline.to(_torch.device(get_device()))
    return _pn_pipeline


# ---------------------------------------------------------------------------
# 主入口：pyannote 优先，异常/缺模型时回退旧管线
# ---------------------------------------------------------------------------

def diarize(y, sr):
    """完整说话人检测入口。

    返回 dict:
        segments: [{start, end, label}]（采样点；label 为说话人编号）
        n_speakers: 说话人数（0 = 无人声）
        total_speech: 语音总时长（秒）
    """
    try:
        pipe = get_pyannote_pipeline()
    except Exception:  # noqa: BLE001
        pipe = None
    if pipe is not None:
        try:
            return _diarize_pyannote(y, sr, pipe)
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
    return _diarize_legacy(y, sr)


def _diarize_pyannote(y, sr, pipe):
    """pyannote 后端：整段音频一次推理，返回与旧后端相同结构。"""
    import torch

    if sr != 16000:
        import librosa

        y16 = librosa.resample(np.asarray(y, dtype="float32"),
                               orig_sr=sr, target_sr=16000)
    else:
        y16 = np.asarray(y, dtype="float32")
    waveform = torch.from_numpy(y16).unsqueeze(0)
    with torch.no_grad():
        annotation = pipe({"waveform": waveform, "sample_rate": 16000})

    # 收集说话人轮次；label 字符串（SPEAKER_00…）按首次出现顺序映射成 0..n-1
    turns = []
    order = {}
    for seg, _, lab in annotation.itertracks(yield_label=True):
        if lab not in order:
            order[lab] = len(order)
        turns.append((seg.start, seg.end, order[lab]))
    turns.sort(key=lambda t: (t[0], t[1]))
    out = [{"start": int(a * sr), "end": int(b * sr), "label": int(l),
            "dur": round(b - a, 2)} for a, b, l in turns]

    def _merge_adjacent(segs):
        """合并相邻同标签段（间隔≤0.5s），消除碎粒。"""
        merged = []
        for s in segs:
            if (merged and s["label"] == merged[-1]["label"]
                    and (s["start"] - merged[-1]["end"]) <= int(0.5 * sr)):
                merged[-1]["end"] = s["end"]
                merged[-1]["dur"] = round(
                    (merged[-1]["end"] - merged[-1]["start"]) / sr, 2)
            else:
                merged.append(dict(s))
        return merged

    # 夹心过滤：夹在同说话人之间的短促误标段（换气/起音被误判成另一人）
    # 归回两侧标签。仅当中间段比两侧邻居都短时才翻转，避免把真实的
    # 短语句误翻；过滤前后各合并一次，碎粒不参与比较
    out = _merge_adjacent(out)
    for _ in range(3):
        changed = False
        for i in range(1, len(out) - 1):
            c, lft, rgt = out[i], out[i - 1], out[i + 1]
            if (lft["label"] == rgt["label"] != c["label"]
                    and c["dur"] < min(lft["dur"], rgt["dur"])):
                c["label"] = lft["label"]
                changed = True
        if not changed:
            break
    out = _merge_adjacent(out)
    total = round(sum(s["dur"] for s in out), 2)
    return {"segments": out, "n_speakers": len(order), "total_speech": total}


# ---------------------------------------------------------------------------
# 旧后端（兜底）：能量 VAD → ECAPA 声纹 → 层次聚类
# ---------------------------------------------------------------------------

def voice_segments(y, sr, min_dur=0.8, max_dur=15.0, silence_gap=0.4, min_rms=0.012):
    """基于能量的 VAD：把连续语音切成 [(start, end)]（采样点），供嵌入/换声使用。

    参数：
        y            单声道 float32 采样点
        sr           采样率
        min_dur      最短语音段（秒），太短不算
        max_dur      单段最长（秒），超出按 max_dur 切开
        silence_gap  静音超过该时长才分段
        min_rms      语音能量阈值
    """
    win = int(sr * 0.03)
    hop = int(sr * 0.01)
    n = len(y)
    if n <= win:
        return [] if n < min_dur * sr else [(0, n)]
    # 滑窗 RMS（cumsum 向量化，等价于逐帧 y[i:i+win].mean()）；
    # 逐帧 Python 循环在长音频上要几十秒，向量化后毫秒级
    x = np.concatenate(([0.0], np.cumsum(np.asarray(y, dtype=np.float64) ** 2)))
    frames = max(1, (n - win) // hop + 1)
    starts = np.arange(frames) * hop
    rms = np.sqrt((x[starts + win] - x[starts]) / win)
    voiced = rms > min_rms
    segs = []
    start = None
    last_voiced = None
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            last_voiced = i
        elif start is not None:
            # 静音持续超过 silence_gap 才分段（按"距最近有声帧的时长"度量，
            # 换气/短停顿不会把一句话切碎）
            if (i - last_voiced) * 0.01 > silence_gap:
                segs.append((start * hop, (last_voiced + 1) * hop))
                start = None
    if start is not None:
        segs.append((start * hop, (last_voiced + 1) * hop))
    # 过滤过短段 + 切分超长段
    out = []
    for a, b in segs:
        if b - a < min_dur * sr:
            continue
        while b - a > max_dur * sr:
            out.append((a, a + int(max_dur * sr)))
            a += int(max_dur * sr)
        if b - a >= min_dur * sr:
            out.append((a, b))
    return out


def _embed(wav16, sr16=16000):
    """一段 16k 单声道语音 → 192 维声纹向量。"""
    import torch

    model = get_model()
    t = torch.from_numpy(np.asarray(wav16, dtype="float32")).unsqueeze(0)
    with torch.no_grad():
        e = model.encode_batch(t).squeeze(0).squeeze(0)
    return e.cpu().numpy()


def segment_embeddings(y, sr, segs):
    """逐段提取声纹向量，返回 (embeds, durations)：
    embeds: n x 192；durations: 每段秒数。"""
    import librosa

    embeds = []
    durations = []
    for a, b in segs:
        seg = y[a:b]
        if len(seg) < int(0.5 * sr):
            continue
        w16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
        embeds.append(_embed(w16))
        durations.append((b - a) / sr)
    return np.asarray(embeds, dtype="float32"), np.asarray(durations, dtype="float32")


def decide_speakers(embeds, durations, min_cluster_ratio=0.12, sil_thr=0.10):
    """聚类判定说话人数。返回 (labels, n_speakers)：
    labels[i] 是第 i 段的说话人编号（0..n-1）。

    规则：
      - 只有 1 段 → 1 人
      - 尝试 k=2、k=3：每簇语音时长占比都要 >= min_cluster_ratio（防止把
        偶尔几句跑调当第二人）；少数说话人必须 ≥2 段，除非其占比 ≥ 0.25
        （避免把 1 句环境音/口误误判成第二人）；k=2 时轮廓系数过低视为
        "其实是一个人"。
    """
    n = len(embeds)
    if n == 0:
        return [], 0
    if n == 1:
        return [0], 1

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    E = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
    if n == 2:
        # 2 段没法用轮廓系数/占比规则，直接比两段声纹相似度：
        # 实测同人段 ≈0.67、异人段 ≈0.15~0.25，阈值取中间偏保守的 0.45
        sim = float(E[0] @ E[1])
        if sim < 0.45:
            return [0, 1], 2
        return [0] * 2, 1
    dist = 1.0 - (E @ E.T)  # 余弦距离矩阵
    for k in (2, 3, 4):  # 最多按 4 个说话人尝试（对话视频常见 3~4 人）
        if n < k + 1:
            break
        cl = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        )
        lab = cl.fit_predict(dist)
        dur = np.zeros(k)
        cnt = np.zeros(k, dtype=int)
        for i, l in enumerate(lab):
            dur[l] += durations[i]
            cnt[l] += 1
        frac = dur / (dur.sum() + 1e-9)
        if frac.min() < min_cluster_ratio:
            continue  # 某簇太少，像噪声/口误，不算独立说话人
        if cnt.min() < 2 and frac.min() < 0.25:
            continue  # 只有 1 段且占比不高 → 判为同一个人
        if k == 2 and n >= 3:
            sil = float(silhouette_score(dist, lab))
            if sil < sil_thr:
                break  # 两簇分得不够开，判为 1 人
        return lab.tolist(), k
    return [0] * n, 1


def _diarize_legacy(y, sr):
    """旧后端完整流程：能量 VAD → ECAPA → 聚类。"""
    segs = voice_segments(y, sr)
    if not segs:
        return {"segments": [], "n_speakers": 0, "total_speech": 0.0}
    embeds, durations = segment_embeddings(y, sr, segs)
    if len(embeds) == 0:
        return {"segments": [], "n_speakers": 0, "total_speech": 0.0}
    labels, k = decide_speakers(embeds, durations)
    out = []
    for i, (a, b) in enumerate(segs[: len(labels)]):
        out.append(
            {"start": int(a), "end": int(b), "label": int(labels[i]),
             "dur": round((b - a) / sr, 2)}
        )
    return {
        "segments": out,
        "n_speakers": k,
        "total_speech": round(float(durations.sum()), 2),
    }


if __name__ == "__main__":
    # 自检：打印测试素材的说话人检测结果
    import sys
    import librosa

    if len(sys.argv) < 2:
        sys.exit("用法: python diarize.py <音频文件>  （对该音频做说话人检测自检）")
    path = sys.argv[1]
    y, sr = librosa.load(path, sr=44100, mono=True)
    r = diarize(y, sr)
    print("n_speakers =", r["n_speakers"], " total_speech =", r["total_speech"])
    for s in r["segments"]:
        print("  [%7.2f - %7.2f] label=%d dur=%.2fs" % (
            s["start"] / sr, s["end"] / sr, s["label"], s["dur"]))
