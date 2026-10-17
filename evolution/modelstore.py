"""Guardado y carga de modelos.

Formato: ``<color>.pth`` es un ``state_dict`` plano de ``SuperTatetiNet`` (igual que
los .pth que ya usan ``play.py`` y ``test_ias.py``), más un archivo ``<color>.json``
con arquitectura, versión y hash de los pesos. Los archivos nunca se sobrescriben.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from model import SuperTatetiNet, hidden_from_state_dict

MODEL_FORMAT_VERSION = 1
ARCHITECTURE_NAME = "SuperTatetiNet"


class ModelExistsError(FileExistsError):
    pass


class ModelIntegrityError(RuntimeError):
    pass


def architecture_signature(state_dict: dict | None = None) -> str:
    """Hash de nombres y formas de los parámetros de la red.

    Sin argumento usa la red por defecto (``SuperTatetiNet()``); con un ``state_dict``
    usa sus formas reales, de modo que la firma refleje el tamaño real del modelo.
    """
    sd = SuperTatetiNet().state_dict() if state_dict is None else state_dict
    shapes = {k: list(v.shape) for k, v in sd.items()}
    return hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest()[:16]


def weights_hash(state_dict: dict) -> str:
    """Hash estable de los pesos (independiente del empaquetado del .pth)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def file_name_for(color: str) -> str:
    return f"{color.lstrip('#')}.pth"


def save_model(path: Path, state_dict: dict, meta: dict) -> str:
    """Escritura atómica y SIN sobrescribir. Devuelve el hash de los pesos."""
    path = Path(path)
    if path.exists() or sidecar_path(path).exists():
        raise ModelExistsError(f"El modelo ya existe y no se sobrescribe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = weights_hash(state_dict)
    info = {
        "format_version": MODEL_FORMAT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "architecture_signature": architecture_signature(state_dict),
        "weights_sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        **meta,
    }
    tmp_w = path.with_name(path.name + ".tmp")
    tmp_j = sidecar_path(path).with_name(sidecar_path(path).name + ".tmp")
    torch.save({k: v.detach().cpu().clone() for k, v in state_dict.items()}, tmp_w)
    tmp_j.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    for tmp in (tmp_w, tmp_j):
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    # El .json se publica primero y el .pth al final: un .pth presente siempre tiene su .json
    os.replace(tmp_j, sidecar_path(path))
    os.replace(tmp_w, path)
    return digest


def load_state_dict(path: Path, expected_sha256: str | None = None) -> dict:
    """Carga los pesos verificando la arquitectura y (si se da) el hash esperado."""
    path = Path(path)
    sd = torch.load(path, weights_only=True, map_location="cpu")
    # Reconstruye la forma real desde el propio state_dict: así checkpoints de otros
    # tamaños (p. ej. viejos 256-256-128) siguen cargando aunque cambie el default.
    try:
        reference = SuperTatetiNet(hidden=hidden_from_state_dict(sd)).state_dict()
    except ValueError:
        reference = SuperTatetiNet().state_dict()
    if set(sd) != set(reference) or any(sd[k].shape != reference[k].shape for k in reference):
        raise ModelIntegrityError(f"Arquitectura incompatible con SuperTatetiNet: {path}")
    if expected_sha256 is not None and weights_hash(sd) != expected_sha256:
        raise ModelIntegrityError(f"El hash de los pesos no coincide: {path}")
    return sd


def load_net(path: Path, expected_sha256: str | None = None) -> SuperTatetiNet:
    sd = load_state_dict(path, expected_sha256)
    try:
        net = SuperTatetiNet(hidden=hidden_from_state_dict(sd))
    except ValueError:
        net = SuperTatetiNet()
    net.load_state_dict(sd)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def copy_model(src: Path, dst: Path) -> None:
    """Copia (sin sobrescribir) un modelo y su .json."""
    if dst.exists():
        raise ModelExistsError(str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sidecar_path(src), sidecar_path(dst))
    shutil.copy2(src, dst)
