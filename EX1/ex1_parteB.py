"""
EXERCÍCIO 1 - ITEM B
Sobreposição de realidade aumentada

O programa utiliza a calibração obtida no Item A para estimar
a pose do tabuleiro em cada frame de um vídeo.

A partir da pose estimada, um cubo 3D virtual com aresta igual
a um quadrado do tabuleiro é projetado sobre a imagem.
"""

from pathlib import Path

import cv2
import numpy as np

from synthetic_utils import (
    generate_dataset,
    calibrate_from_paths,
    find_refined_corners,
    object_points,
    SQUARE_SIZE_M
)


ROOT = Path(__file__).resolve().parent


# ==========================================================
# ETAPA 1 - Carregar a calibração do Item A
# ==========================================================

paths = generate_dataset(
    ROOT,
    n=18
)

(
    _,
    K,
    dist,
    _,
    _,
    _,
    _,
    _,
    _
) = calibrate_from_paths(paths)

print("=== CALIBRAÇÃO CARREGADA ===")

print("\nMatriz K:")
print(K)

print("\nCoeficientes de distorção:")
print(dist.ravel()[:5])


# ==========================================================
# ETAPA 2 - Definir o cubo 3D
# ==========================================================

s = SQUARE_SIZE_M

cubo = np.float32([
    [0, 0, 0],
    [s, 0, 0],
    [s, s, 0],
    [0, s, 0],

    [0, 0, -s],
    [s, 0, -s],
    [s, s, -s],
    [0, s, -s]
])


# ==========================================================
# ETAPA 3 - Abrir o vídeo
# ==========================================================

video = (
    ROOT
    / "dados_sinteticos"
    / "video_tabuleiro.avi"
)

cap = cv2.VideoCapture(
    str(video)
)

if not cap.isOpened():
    raise RuntimeError(
        "Não foi possível abrir o vídeo."
    )


# ==========================================================
# ETAPA 4 - Preparar vídeo de saída
# ==========================================================

saida = (
    ROOT
    / "saidas"
    / "video_ar.avi"
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


# ==========================================================
# ETAPA 5 - Processamento dos frames
# ==========================================================

try:
    while True:

        ok_frame, frame = cap.read()

        if not ok_frame:
            break

        # Detecta e refina os cantos do tabuleiro.
        ok, corners = find_refined_corners(
            frame
        )

        if ok:

            # Estima a pose do tabuleiro.
            sucesso, rvec, tvec = cv2.solvePnP(
                object_points(),
                corners,
                K,
                dist
            )

            if sucesso:

                # Projeta os pontos 3D do cubo
                # para coordenadas 2D da imagem.
                pts, _ = cv2.projectPoints(
                    cubo,
                    rvec,
                    tvec,
                    K,
                    dist
                )

                p = (
                    pts
                    .reshape(-1, 2)
                    .astype(int)
                )

                # ==========================================
                # Base do cubo - verde
                # ==========================================

                for a, b in [
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 0)
                ]:
                    cv2.line(
                        frame,
                        tuple(p[a]),
                        tuple(p[b]),
                        (0, 255, 0),
                        3
                    )

                # ==========================================
                # Topo do cubo - azul
                # ==========================================

                for a, b in [
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 4)
                ]:
                    cv2.line(
                        frame,
                        tuple(p[a]),
                        tuple(p[b]),
                        (255, 0, 0),
                        3
                    )

                # ==========================================
                # Arestas verticais - vermelho
                # ==========================================

                for a, b in [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7)
                ]:
                    cv2.line(
                        frame,
                        tuple(p[a]),
                        tuple(p[b]),
                        (0, 0, 255),
                        3
                    )

                # ==========================================
                # Pose estimada
                # ==========================================

                print("\nrvec:")
                print(rvec.ravel())

                print("tvec [m]:")
                print(tvec.ravel())

        # Salva o frame no vídeo de saída.
        writer.write(frame)

        # Reduz apenas para exibição.
        frame_exibicao = cv2.resize(
            frame,
            None,
            fx=0.75,
            fy=0.75,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow(
            "Realidade Aumentada",
            frame_exibicao
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


print(
    "\nVídeo com realidade aumentada salvo em:",
    saida
)