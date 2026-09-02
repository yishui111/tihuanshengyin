# -*- coding: utf-8 -*-
"""
角色注册表：把 A(RVC) / C(SoVITS) / D(GPT-SoVITS) 的角色合并成统一列表
=====================================================================
批量换声页面用一个下拉就能选到所有角色，自动路由到对应引擎服务。
引擎 B(OpenVoice) 需要参考人声，不参与角色表（单文件模式仍可用）。
"""

import os
import socket

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 各引擎服务端口（与各自 bat 默认一致）
PORTS = {"A": 8010, "B": 8020, "C": 8030, "D": 8040}

# 引擎显示名
ENGINE_CN = {"A": "RVC", "B": "OpenVoice", "C": "SoVITS", "D": "GPT-SoVITS"}

# 已知角色的中文名（引擎内找不到中文名时用英文名兜底）
CN_NAMES = {
    # A：RVC 模型（rvc/assets/weights/*.pth）
    "Nahida": "纳西妲",
    "NahidaCN": "纳西妲2",
    "Paimon": "派蒙",
    "dabing": "大彬",
    "liejun": "列军",
    "songwukong": "孙悟空",
    "testnv": "测试女声",
    # C：SoVITS 中配角色（sovits_service/models/*.json）
    "nahida": "纳西妲",
    "klee": "可莉",
    "hutao": "胡桃",
    "yaoyao": "瑶瑶",
    "raiden": "雷电将军",
    "furina": "芙宁娜",
    # D：GPT-SoVITS 角色（gptsovits_service/models/*/）
    "ayaka": "神里绫华",
    "azhong": "Azhong",
}

# C 服务（SoVITS）模型文件名 → 接口角色名对照。
# 模型文件命名（nahida41_G_*.pth / randenEi_G_*.pth）与接口角色名
# （sovits_cn_api.py 的 CHARACTERS 键：nahida / raiden）不一致，需换算。
C_CHAR_OVERRIDES = {"nahida41": "nahida", "randenEi": "raiden"}


def _port_open(port, timeout=0.4):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def load_roles():
    """扫描各引擎模型目录，返回统一角色列表。

    每条：{id, engine, engine_cn, port, character, name, online}
    id 形如 "C:klee"，用于页面下拉和批量任务映射。
    """
    roles = []

    # ---- A：RVC ----
    wdir = os.path.join(PROJECT_ROOT, "rvc", "assets", "weights")
    if os.path.isdir(wdir):
        for f in sorted(os.listdir(wdir)):
            if f.endswith(".pth"):
                name = f[:-4]
                roles.append({
                    "id": "A:" + name, "engine": "A",
                    "engine_cn": ENGINE_CN["A"], "port": PORTS["A"],
                    "character": name, "name": CN_NAMES.get(name, name),
                })

    # ---- C：SoVITS（角色 json 配置）----
    mdir = os.path.join(PROJECT_ROOT, "sovits_service", "models")
    if os.path.isdir(mdir):
        for f in sorted(os.listdir(mdir)):
            if f.endswith(".json"):
                name = C_CHAR_OVERRIDES.get(os.path.splitext(f)[0], os.path.splitext(f)[0])
                roles.append({
                    "id": "C:" + name, "engine": "C",
                    "engine_cn": ENGINE_CN["C"], "port": PORTS["C"],
                    "character": name, "name": CN_NAMES.get(name, name),
                })

    # ---- D：GPT-SoVITS（每个角色一个子目录）----
    gdir = os.path.join(PROJECT_ROOT, "gptsovits_service", "models")
    if os.path.isdir(gdir):
        for name in sorted(os.listdir(gdir)):
            sub = os.path.join(gdir, name)
            if os.path.isdir(sub) and any(
                x.endswith((".pth", ".ckpt")) for x in os.listdir(sub)
            ):
                roles.append({
                    "id": "D:" + name, "engine": "D",
                    "engine_cn": ENGINE_CN["D"], "port": PORTS["D"],
                    "character": name, "name": CN_NAMES.get(name, name),
                })

    # 在线状态统一刷新
    for r in roles:
        r["online"] = _port_open(r["port"])
    return roles


def role_by_id(role_id, roles=None):
    roles = roles if roles is not None else load_roles()
    for r in roles:
        if r["id"] == role_id:
            return r
    return None
