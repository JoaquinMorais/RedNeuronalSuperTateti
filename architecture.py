"""Fuente única de verdad de la arquitectura de las redes del proyecto.

Cambiar la estructura de una red = editar UNA línea de este archivo. Todo el repo
(entrenamiento por puzzles, ``play.py``, la liga evolutiva y ``evolve_puzzles``) lee
estos valores para construir redes NUEVAS.

Importante: el tamaño "real" de un modelo entrenado vive en su propio ``state_dict`` y
se infiere al cargarlo (``model.hidden_from_state_dict``). Estos valores son solo el
default para redes frescas; por eso subir o bajar la capacidad no rompe checkpoints
viejos (siguen cargando por inferencia).
"""
from __future__ import annotations

# MLP (``model.SuperTatetiNet``): dimensiones de las capas ocultas, de entrada a salida.
# 384-384-256 (~322k params) frente a 256-256-128 (~143k params).
MLP_HIDDEN = (384, 384, 256)

# CNN (``trainPuzzles.model_cnn.SuperTatetiCNN``): canales del cuerpo convolucional y
# tamaño del kernel de sus convoluciones (p. ej. (64, 128, 128) para ~240k params).
CNN_CHANNELS = (48, 96, 96)
CNN_KERNEL_SIZE = 3
