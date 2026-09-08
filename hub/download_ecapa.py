# -*- coding: utf-8 -*-
"""
ECAPA 声纹模型下载/补缓存工具（一次性联网使用，之后 hub 全离线可跑）
====================================================================
作用：把说话人检测用的 speechbrain 声纹模型
      speechbrain/spkrec-ecapa-voxceleb 下载到本地缓存：
        runtime\\cache\\hf_speaker_model     （模型参数，savedir）
        runtime\\cache\\huggingface\\hub     （speechbrain/hf 缓存）
用法（需能联网；走代理时先设好 HTTPS_PROXY）：
    runtime\\py312\\python.exe hub\\download_ecapa.py
联网要求见 DEPLOY.md §6.5；下载成功后 "说话人排查" 全离线可用。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(PROJECT_ROOT, "runtime", "cache", "huggingface")
SPK_SAVEDIR = os.path.join(PROJECT_ROOT, "runtime", "cache", "hf_speaker_model")


def main():
    # 与 hub/diarize.py::get_model() 完全一致的加载参数（去掉强制离线），
    # 确保下载产物能被 diarize 的离线加载原样命中。
    os.environ["HF_HOME"] = HF_CACHE
    # 不要设 HF_HUB_OFFLINE=1：本工具必须联网
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    print("downloading speechbrain/spkrec-ecapa-voxceleb ...")
    print("  HF_HOME      =", HF_CACHE)
    print("  savedir      =", SPK_SAVEDIR)
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=SPK_SAVEDIR,
        local_strategy=LocalStrategy.COPY,
        run_opts={"device": "cpu"},
    )
    # 加载成功即触发超参/权重的完整下载与落盘
    print("download OK: model loaded")

    # 打印落盘文件清单，便于人工核对
    for base, label in ((SPK_SAVEDIR, "savedir"), (HF_CACHE, "hf_cache")):
        if os.path.isdir(base):
            files = []
            for root, _dirs, names in os.walk(base):
                for n in names:
                    p = os.path.join(root, n)
                    if os.path.isfile(p):
                        files.append((p, os.path.getsize(p)))
            files.sort()
            total = sum(s for _, s in files) // (1024 * 1024)
            print("[%s] %d files, ~%d MB:" % (label, len(files), total))
            for p, s in files[:40]:
                print("   %10d  %s" % (s, p))
            if len(files) > 40:
                print("   ... (%d more)" % (len(files) - 40))
    return 0


if __name__ == "__main__":
    sys.exit(main())
