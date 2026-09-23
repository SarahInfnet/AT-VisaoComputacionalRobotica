from pathlib import Path
import time
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================================
# CAMINHOS DO PROJETO
# ==========================================================

ROOT = Path(__file__).resolve().parent

VIDEO = ROOT / "dados" / "video_real.mp4"

PASTA_SAIDAS = ROOT / "saidas"
PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)

VIDEO_SAIDA = PASTA_SAIDAS / "video_yolo_tracking.avi"


# ==========================================================
# PARÂMETROS
# ==========================================================

CONFIANCA = 0.25
NMS_THRESHOLD = 0.40

IOU_THRESHOLD = 0.30

TAMANHO_TRILHA = 30

ESCALA_EXIBICAO = 0.60


# ==========================================================
# FUNÇÃO IoU
# ==========================================================

def calcular_iou(caixa_a, caixa_b):

    x1_a, y1_a, x2_a, y2_a = caixa_a
    x1_b, y1_b, x2_b, y2_b = caixa_b

    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)

    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)

    largura_inter = max(0, x2_inter - x1_inter)
    altura_inter = max(0, y2_inter - y1_inter)

    area_inter = largura_inter * altura_inter

    area_a = (
        (x2_a - x1_a) *
        (y2_a - y1_a)
    )

    area_b = (
        (x2_b - x1_b) *
        (y2_b - y1_b)
    )

    area_uniao = area_a + area_b - area_inter

    if area_uniao <= 0:
        return 0.0

    return area_inter / area_uniao


# ==========================================================
# COR DO ID
# ==========================================================

def gerar_cor(id_objeto):

    cor = (
        int((id_objeto * 37) % 255),
        int((id_objeto * 67) % 255),
        int((id_objeto * 97) % 255)
    )

    return cor


# ==========================================================
# CENTRO DA CAIXA
# ==========================================================

def calcular_centro(caixa):

    x1, y1, x2, y2 = caixa

    centro_x = int((x1 + x2) / 2)
    centro_y = int((y1 + y2) / 2)

    return centro_x, centro_y


# ==========================================================
# PROCESSAMENTO
# ==========================================================

def main():

    # ------------------------------------------------------
    # Verificação do vídeo
    # ------------------------------------------------------

    if not VIDEO.exists():
        print("Vídeo não encontrado:")
        print(VIDEO)
        return

    # ------------------------------------------------------
    # Carregamento do YOLO
    # ------------------------------------------------------

    print("\n==========================================")
    print("YOLOv8n + RASTREAMENTO POR IoU")
    print("==========================================")

    modelo = YOLO("yolov8n.pt")

    captura = cv2.VideoCapture(str(VIDEO))

    if not captura.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    largura = int(
        captura.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    altura = int(
        captura.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps_video = captura.get(
        cv2.CAP_PROP_FPS
    )

    total_frames_video = int(
        captura.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps_video <= 0:
        fps_video = 30.0

    duracao_segundos = (
        total_frames_video / fps_video
    )

    # ------------------------------------------------------
    # Vídeo de saída
    # ------------------------------------------------------

    codec = cv2.VideoWriter_fourcc(*"MJPG")

    escritor = cv2.VideoWriter(
        str(VIDEO_SAIDA),
        codec,
        fps_video,
        (largura, altura)
    )

    # ------------------------------------------------------
    # Estruturas do rastreamento
    # ------------------------------------------------------

    objetos_anteriores = {}

    trilhas = {}

    proximo_id = 1

    entradas = 0
    saidas = 0

    id_switches = 0

    total_frames = 0

    latencias = []

    # ------------------------------------------------------
    # Processamento frame a frame
    # ------------------------------------------------------

    while True:

        ok, frame = captura.read()

        if not ok:
            break

        total_frames += 1

        # --------------------------------------------------
        # Detecção YOLO
        # --------------------------------------------------

        inicio = time.perf_counter()

        resultado = modelo(
            frame,
            conf=CONFIANCA,
            iou=NMS_THRESHOLD,
            verbose=False
        )[0]

        fim = time.perf_counter()

        latencia = (fim - inicio) * 1000
        latencias.append(latencia)

        # --------------------------------------------------
        # Detecções do frame atual
        # --------------------------------------------------

        deteccoes_atuais = []

        for caixa in resultado.boxes:

            coordenadas = (
                caixa.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordenadas

            confianca = float(
                caixa.conf[0]
            )

            id_classe = int(
                caixa.cls[0]
            )

            nome_classe = modelo.names[
                id_classe
            ]

            deteccoes_atuais.append(
                {
                    "caixa": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "classe": nome_classe,
                    "confianca": confianca,
                    "id": None
                }
            )

        # --------------------------------------------------
        # Associação por IoU
        # --------------------------------------------------

        ids_usados = set()

        for deteccao in deteccoes_atuais:

            melhor_iou = 0.0
            melhor_id = None

            for id_objeto, objeto_anterior in objetos_anteriores.items():

                if id_objeto in ids_usados:
                    continue

                # Só compara objetos da mesma classe
                if (
                    deteccao["classe"]
                    != objeto_anterior["classe"]
                ):
                    continue

                iou = calcular_iou(
                    deteccao["caixa"],
                    objeto_anterior["caixa"]
                )

                if iou > melhor_iou:
                    melhor_iou = iou
                    melhor_id = id_objeto

            # ----------------------------------------------
            # ID persistente
            # ----------------------------------------------

            if (
                melhor_id is not None
                and melhor_iou >= IOU_THRESHOLD
            ):

                deteccao["id"] = melhor_id
                ids_usados.add(melhor_id)

            else:

                deteccao["id"] = proximo_id

                trilhas[proximo_id] = []

                proximo_id += 1

                entradas += 1

        # --------------------------------------------------
        # Objetos que saíram
        # --------------------------------------------------

        ids_atuais = {
            deteccao["id"]
            for deteccao in deteccoes_atuais
        }

        for id_anterior in objetos_anteriores:

            if id_anterior not in ids_atuais:
                saidas += 1

        # --------------------------------------------------
        # Estimativa de ID switches
        # --------------------------------------------------

        for deteccao in deteccoes_atuais:

            melhor_iou = 0.0
            id_anterior_mais_proximo = None

            for id_anterior, objeto_anterior in objetos_anteriores.items():

                if (
                    deteccao["classe"]
                    != objeto_anterior["classe"]
                ):
                    continue

                iou = calcular_iou(
                    deteccao["caixa"],
                    objeto_anterior["caixa"]
                )

                if iou > melhor_iou:
                    melhor_iou = iou
                    id_anterior_mais_proximo = id_anterior

            if (
                id_anterior_mais_proximo is not None
                and melhor_iou >= IOU_THRESHOLD
                and deteccao["id"]
                != id_anterior_mais_proximo
            ):
                id_switches += 1

        # --------------------------------------------------
        # Atualização das trilhas
        # --------------------------------------------------

        for deteccao in deteccoes_atuais:

            id_objeto = deteccao["id"]

            centro = calcular_centro(
                deteccao["caixa"]
            )

            if id_objeto not in trilhas:
                trilhas[id_objeto] = []

            trilhas[id_objeto].append(
                centro
            )

            if (
                len(trilhas[id_objeto])
                > TAMANHO_TRILHA
            ):
                trilhas[id_objeto].pop(0)

        # --------------------------------------------------
        # Desenho das caixas, IDs e trilhas
        # --------------------------------------------------

        for deteccao in deteccoes_atuais:

            x1, y1, x2, y2 = (
                deteccao["caixa"]
            )

            id_objeto = deteccao["id"]

            nome_classe = (
                deteccao["classe"]
            )

            confianca = (
                deteccao["confianca"]
            )

            cor = gerar_cor(
                id_objeto
            )

            # Caixa
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                cor,
                2
            )

            texto = (
                f"ID {id_objeto} "
                f"{nome_classe} "
                f"{confianca * 100:.1f}%"
            )

            cv2.putText(
                frame,
                texto,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cor,
                2
            )

            # ----------------------------------------------
            # Trilha dos últimos 30 frames
            # ----------------------------------------------

            pontos = trilhas[
                id_objeto
            ]

            for i in range(
                1,
                len(pontos)
            ):

                cv2.line(
                    frame,
                    pontos[i - 1],
                    pontos[i],
                    cor,
                    2
                )

        # --------------------------------------------------
        # Contagem na tela
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"Entradas: {entradas}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Saidas: {saidas}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Objetos atuais: {len(deteccoes_atuais)}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # --------------------------------------------------
        # Atualização para o próximo frame
        # --------------------------------------------------

        objetos_anteriores = {}

        for deteccao in deteccoes_atuais:

            objetos_anteriores[
                deteccao["id"]
            ] = {
                "caixa": deteccao["caixa"],
                "classe": deteccao["classe"]
            }

        # --------------------------------------------------
        # Gravação
        # --------------------------------------------------

        escritor.write(frame)

        # --------------------------------------------------
        # Exibição reduzida
        # --------------------------------------------------

        frame_exibicao = cv2.resize(
            frame,
            None,
            fx=ESCALA_EXIBICAO,
            fy=ESCALA_EXIBICAO
        )

        cv2.imshow(
            "YOLOv8n + Tracking IoU",
            frame_exibicao
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    # ------------------------------------------------------
    # Finalização
    # ------------------------------------------------------

    captura.release()
    escritor.release()
    cv2.destroyAllWindows()

    # ------------------------------------------------------
    # Cálculo das métricas
    # ------------------------------------------------------

    if len(latencias) > 0:

        latencia_media = np.mean(
            latencias
        )

        fps_medio = (
            1000 / latencia_media
        )

    else:

        latencia_media = 0
        fps_medio = 0

    if duracao_segundos > 0:

        duracao_minutos = (
            duracao_segundos / 60
        )

        taxa_id_switches = (
            id_switches
            / duracao_minutos
        )

    else:

        taxa_id_switches = 0

    # ------------------------------------------------------
    # Resultados
    # ------------------------------------------------------

    print("\n==========================================")
    print("RESULTADOS DO RASTREAMENTO")
    print("==========================================")

    print(
        f"Frames processados: "
        f"{total_frames}"
    )

    print(
        f"Entradas acumuladas: "
        f"{entradas}"
    )

    print(
        f"Saídas acumuladas: "
        f"{saidas}"
    )

    print(
        f"ID switches: "
        f"{id_switches}"
    )

    print(
        f"Taxa de ID switches por minuto: "
        f"{taxa_id_switches:.2f}"
    )

    print(
        f"Latência média YOLO: "
        f"{latencia_media:.2f} ms"
    )

    print(
        f"FPS médio estimado: "
        f"{fps_medio:.2f}"
    )

    print("\nVídeo gerado:")
    print(VIDEO_SAIDA)


# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    main()