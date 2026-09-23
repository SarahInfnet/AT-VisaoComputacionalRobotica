from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ==========================================================
# CAMINHOS DO PROJETO
# ==========================================================

ROOT = Path(__file__).resolve().parent

VIDEO = ROOT / "dados" / "video_real.mp4"

SSD_PB = ROOT / "modelos" / "frozen_inference_graph.pb"
SSD_PBTXT = ROOT / "modelos" / "ssd_mobilenet_v2_coco.pbtxt"

PASTA_SAIDAS = ROOT / "saidas"
PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)

VIDEO_YOLO = PASTA_SAIDAS / "video_yolo.avi"
VIDEO_SSD = PASTA_SAIDAS / "video_ssd.avi"


# ==========================================================
# PARÂMETROS
# ==========================================================

CONFIANCA = 0.25
NMS_THRESHOLD = 0.40

# Apenas para diminuir a janela mostrada na tela.
# Não altera o frame processado nem o vídeo salvo.
ESCALA_EXIBICAO = 0.60


# ==========================================================
# CLASSES COCO PARA SSD MOBILENETV2
# ==========================================================

CLASSES_COCO = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]


# ==========================================================
# FUNÇÃO PARA TAMANHO DO ARQUIVO
# ==========================================================

def tamanho_mb(caminho):
    if caminho.exists():
        return caminho.stat().st_size / (1024 * 1024)

    return None


# ==========================================================
# PROCESSAMENTO COM YOLOv8n
# ==========================================================

def processar_yolo():

    print("\n==========================================")
    print("PROCESSANDO YOLOv8n")
    print("==========================================")

    modelo = YOLO("yolov8n.pt")

    captura = cv2.VideoCapture(str(VIDEO))

    if not captura.isOpened():
        print("Erro ao abrir o vídeo.")
        return None, None

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = captura.get(cv2.CAP_PROP_FPS)

    if fps_video <= 0:
        fps_video = 30.0

    codec = cv2.VideoWriter_fourcc(*"MJPG")

    escritor = cv2.VideoWriter(
        str(VIDEO_YOLO),
        codec,
        fps_video,
        (largura, altura)
    )

    latencias = []
    total_frames = 0

    while True:

        ok, frame = captura.read()

        if not ok:
            break

        # --------------------------------------------------
        # Inferência YOLO
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
        # Desenho das detecções
        # --------------------------------------------------

        frame_anotado = resultado.plot()

        escritor.write(frame_anotado)

        # --------------------------------------------------
        # Redimensionamento SOMENTE para exibição
        # --------------------------------------------------

        frame_exibicao = cv2.resize(
            frame_anotado,
            None,
            fx=ESCALA_EXIBICAO,
            fy=ESCALA_EXIBICAO
        )

        cv2.imshow("YOLOv8n", frame_exibicao)

        total_frames += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    escritor.release()
    cv2.destroyAllWindows()

    if len(latencias) == 0:
        print("Nenhum frame foi processado pelo YOLO.")
        return None, None

    latencia_media = np.mean(latencias)
    fps_medio = 1000 / latencia_media

    print("\n=== RESULTADO YOLOv8n ===")
    print(f"Frames processados: {total_frames}")
    print(f"Latência média: {latencia_media:.2f} ms")
    print(f"FPS médio estimado: {fps_medio:.2f}")

    return fps_medio, latencia_media


# ==========================================================
# PROCESSAMENTO COM SSD MOBILENETV2
# ==========================================================

def processar_ssd():

    print("\n==========================================")
    print("PROCESSANDO SSD MobileNetV2")
    print("==========================================")

    rede = cv2.dnn.readNetFromTensorflow(
        str(SSD_PB),
        str(SSD_PBTXT)
    )

    captura = cv2.VideoCapture(str(VIDEO))

    if not captura.isOpened():
        print("Erro ao abrir o vídeo.")
        return None, None

    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = captura.get(cv2.CAP_PROP_FPS)

    if fps_video <= 0:
        fps_video = 30.0

    codec = cv2.VideoWriter_fourcc(*"MJPG")

    escritor = cv2.VideoWriter(
        str(VIDEO_SSD),
        codec,
        fps_video,
        (largura, altura)
    )

    latencias = []
    total_frames = 0

    while True:

        ok, frame = captura.read()

        if not ok:
            break

        # --------------------------------------------------
        # Preparação da entrada
        # --------------------------------------------------

        blob = cv2.dnn.blobFromImage(
            frame,
            size=(300, 300),
            swapRB=True,
            crop=False
        )

        rede.setInput(blob)

        # --------------------------------------------------
        # Inferência SSD
        # --------------------------------------------------

        inicio = time.perf_counter()

        deteccoes = rede.forward()

        fim = time.perf_counter()

        latencia = (fim - inicio) * 1000
        latencias.append(latencia)

        # --------------------------------------------------
        # Coleta das detecções
        # --------------------------------------------------

        caixas = []
        confiancas = []
        ids_classes = []

        for i in range(deteccoes.shape[2]):

            confianca = float(deteccoes[0, 0, i, 2])

            if confianca >= CONFIANCA:

                id_classe = int(deteccoes[0, 0, i, 1])

                x1 = int(
                    deteccoes[0, 0, i, 3] * largura
                )

                y1 = int(
                    deteccoes[0, 0, i, 4] * altura
                )

                x2 = int(
                    deteccoes[0, 0, i, 5] * largura
                )

                y2 = int(
                    deteccoes[0, 0, i, 6] * altura
                )

                largura_caixa = x2 - x1
                altura_caixa = y2 - y1

                caixas.append(
                    [
                        x1,
                        y1,
                        largura_caixa,
                        altura_caixa
                    ]
                )

                confiancas.append(confianca)
                ids_classes.append(id_classe)

        # --------------------------------------------------
        # Non-Maximum Suppression
        # --------------------------------------------------

        indices = cv2.dnn.NMSBoxes(
            caixas,
            confiancas,
            CONFIANCA,
            NMS_THRESHOLD
        )

        # --------------------------------------------------
        # Desenho das caixas
        # --------------------------------------------------

        if len(indices) > 0:

            for indice in indices:

                indice = int(indice)

                x, y, w, h = caixas[indice]

                confianca = confiancas[indice]
                id_classe = ids_classes[indice]

                if 0 <= id_classe < len(CLASSES_COCO):
                    nome_classe = CLASSES_COCO[id_classe]
                else:
                    nome_classe = f"classe {id_classe}"

                texto = (
                    f"{nome_classe}: "
                    f"{confianca * 100:.1f}%"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    texto,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        escritor.write(frame)

        # --------------------------------------------------
        # Redimensionamento SOMENTE para exibição
        # --------------------------------------------------

        frame_exibicao = cv2.resize(
            frame,
            None,
            fx=ESCALA_EXIBICAO,
            fy=ESCALA_EXIBICAO
        )

        cv2.imshow(
            "SSD MobileNetV2",
            frame_exibicao
        )

        total_frames += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    escritor.release()
    cv2.destroyAllWindows()

    if len(latencias) == 0:
        print("Nenhum frame foi processado pelo SSD.")
        return None, None

    latencia_media = np.mean(latencias)
    fps_medio = 1000 / latencia_media

    print("\n=== RESULTADO SSD MobileNetV2 ===")
    print(f"Frames processados: {total_frames}")
    print(f"Latência média: {latencia_media:.2f} ms")
    print(f"FPS médio estimado: {fps_medio:.2f}")

    return fps_medio, latencia_media


# ==========================================================
# FUNÇÃO PRINCIPAL
# ==========================================================

def main():

    # ------------------------------------------------------
    # Verificação dos arquivos
    # ------------------------------------------------------

    if not VIDEO.exists():
        print("Vídeo não encontrado:")
        print(VIDEO)
        return

    if not SSD_PB.exists():
        print("Modelo SSD não encontrado:")
        print(SSD_PB)
        return

    if not SSD_PBTXT.exists():
        print("Arquivo de configuração SSD não encontrado:")
        print(SSD_PBTXT)
        return

    # ------------------------------------------------------
    # YOLOv8n
    # ------------------------------------------------------

    fps_yolo, latencia_yolo = processar_yolo()

    if fps_yolo is None:
        return

    # ------------------------------------------------------
    # SSD MobileNetV2
    # ------------------------------------------------------

    fps_ssd, latencia_ssd = processar_ssd()

    if fps_ssd is None:
        return

    # ------------------------------------------------------
    # Tamanho dos modelos
    # ------------------------------------------------------

    caminho_yolo = Path("yolov8n.pt")

    tamanho_yolo = tamanho_mb(caminho_yolo)
    tamanho_ssd = tamanho_mb(SSD_PB)

    # ------------------------------------------------------
    # Tabela comparativa
    # ------------------------------------------------------

    dados = [
        {
            "modelo": "YOLOv8n",
            "fps": round(fps_yolo, 2),
            "latencia_ms": round(latencia_yolo, 2),
            "parametros_M": 3.2,
            "tamanho_MB": (
                round(tamanho_yolo, 2)
                if tamanho_yolo is not None
                else None
            )
        },
        {
            "modelo": "SSD MobileNetV2",
            "fps": round(fps_ssd, 2),
            "latencia_ms": round(latencia_ssd, 2),
            "parametros_M": 15.29,
            "tamanho_MB": (
                round(tamanho_ssd, 2)
                if tamanho_ssd is not None
                else None
            )
        }
    ]

    tabela = pd.DataFrame(dados)

    print("\n==========================================")
    print("TABELA COMPARATIVA")
    print("==========================================")
    print(tabela.to_string(index=False))

    # ------------------------------------------------------
    # Arquivos gerados
    # ------------------------------------------------------

    print("\n==========================================")
    print("ARQUIVOS GERADOS")
    print("==========================================")

    print(VIDEO_YOLO)
    print(VIDEO_SSD)


# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":
    main()