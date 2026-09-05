# -*- coding: utf-8 -*-
"""
说话人检测（说话人分离）：VAD 分段 → ECAPA 声纹嵌入 → 层次聚类
===============================================================
用途：分析一段音频里有几个说话人，并给每段语音打上说话人标签，
供批量换声时"按说话人分配角色"使用。

依赖（runtime\\py312 已内置）：
    librosa / numpy / torch / speechbrain(ECAPA, 离线缓存) / scikit-learn
离线模型：
    runtime\\cache\\hf_speaker_model          （ECAPA 声纹模型参数）
    runtime\\cache\\huggingface\\hub\\...     （speechbrain 缓存，防止联网）
"""

import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "runtime", "cache", "huggingface")
SPK_SAVEDIR = os.path.join(PROJECT_ROOT, "runtime", "cache", "hf_speaker_model")

# 模型进程内只加载一次（懒加载）
_model = None
_device = None


def get_device():
    global _device
    if _device is None:
        import torch

        _device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    return _device


def get_model():
    """ECAPA-TDNN 声纹模型（离线加载，进程内单例）。"""
    global _model
    if _model is None:
        os.environ.setdefault("HF_HOME", HF_CACHE)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy

        _model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=SPK_SAVEDIR,
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": get_device()},
        )
    return _model


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


def diarize(y, sr):
    """完整说话人检测入口。

    返回 dict:
        segments: [{start, end, label}]（采样点；label 为说话人编号）
        n_speakers: 说话人数（0 = 无人声）
        total_speech: 语音总时长（秒）
    """
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
