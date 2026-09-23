"""
EXERCÍCIO 1 - ITEM B
Geração do vídeo sintético do tabuleiro

Este programa gera um vídeo com o tabuleiro em movimento,
variando posição, rotação e distância ao longo dos frames.
"""

from pathlib import Path

import cv2
import numpy as np

from synthetic_utils import generate_view


ROOT = Path(__file__).resolve().parent

saida = (
    ROOT
    / "dados_sinteticos"
    / "video_tabuleiro.avi"
)

saida.parent.mkdir(
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    str(saida),
    fourcc,
    15.0,
    (1280, 720)
)

for i in range(90):
    a = i / 89.0

    rx = 8 + 8 * np.sin(a * np.pi * 2)
    ry = -12 + 12 * np.sin(a * np.pi)
    rz = 6 * np.sin(a * np.pi * 2)

    tx = -0.10 + 0.02 * np.sin(a * np.pi * 2)
    ty = -0.07
    tz = 0.75 + 0.05 * np.sin(a * np.pi * 2)

    frame, _, _ = generate_view(
        rx,
        ry,
        rz,
        tx,
        ty,
        tz
    )

    writer.write(frame)

writer.release()

print(
    "Vídeo sintético criado em:",
    saida
)