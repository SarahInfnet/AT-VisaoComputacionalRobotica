"""
Exercicio 2 - Item B
Pipeline integrado de Visao Computacional.

Etapas:
1. Carregar frame e aplicar undistort.
2. Segmentar ROI por cor HSV.
3. Extrair features ORB.
4. Aplicar detector Haar Cascade.
5. Classificar a ROI com OpenCV DNN.
6. Exibir os tempos de cada etapa.
"""

from pathlib import Path
import time

import cv2
import numpy as np

from dnn_utils import (
    get_opencv_net,
    predict_opencv,
    topk,
    load_labels
)


# ==========================================================
# CAMINHOS
# ==========================================================

ROOT = Path(__file__).resolve().parent

arquivo_frame = (
    ROOT
    / "data"
    / "pipeline"
    / "frame_teste.jpg"
)

arquivo_calibracao = (
    ROOT.parent
    / "EX1"
    / "saidas"
    / "calibracao_camera.npz"
)

arquivo_saida = (
    ROOT
    / "saidas"
    / "frame_pipeline_final.jpg"
)

arquivo_undistort = (
    ROOT
    / "saidas"
    / "frame_undistort.jpg"
)


# ==========================================================
# VERIFICAR ARQUIVOS
# ==========================================================

if not arquivo_frame.exists():
    raise FileNotFoundError(
        f"Frame nao encontrado: {arquivo_frame}"
    )

if not arquivo_calibracao.exists():
    raise FileNotFoundError(
        f"Calibracao nao encontrada: "
        f"{arquivo_calibracao}"
    )


# ==========================================================
# CARREGAR FRAME
# ==========================================================

frame = cv2.imread(
    str(arquivo_frame)
)

if frame is None:
    raise RuntimeError(
        "Nao foi possivel abrir o frame."
    )


# ==========================================================
# CARREGAR CALIBRACAO DO EXERCICIO 1
# ==========================================================

dados = np.load(
    str(arquivo_calibracao)
)

K = dados["K"]
dist = dados["dist"]


# ==========================================================
# CARREGAR MODELO DNN
# ==========================================================

net = get_opencv_net()
labels = load_labels()


# ==========================================================
# ETAPA 1 - UNDISTORT
# ==========================================================

inicio = time.perf_counter()

frame_corrigido = cv2.undistort(
    frame,
    K,
    dist
)

fim = time.perf_counter()

tempo_undistort = (
    fim - inicio
) * 1000


# A calibracao do Exercicio 1 foi obtida com imagens
# sinteticas. O resultado do undistort e salvo
# separadamente para demonstrar a etapa de calibracao.
#
# Como o frame utilizado neste exercicio e uma imagem real
# e nao foi capturado pela camera calibrada no Exercicio 1,
# as demais etapas utilizam o frame original.

arquivo_saida.parent.mkdir(
    exist_ok=True
)

cv2.imwrite(
    str(arquivo_undistort),
    frame_corrigido
)

frame_pipeline = frame.copy()


# ==========================================================
# ETAPA 2 - SEGMENTACAO HSV
# ==========================================================

inicio = time.perf_counter()

hsv = cv2.cvtColor(
    frame_pipeline,
    cv2.COLOR_BGR2HSV
)

# O vermelho aparece nas duas extremidades
# do intervalo de matiz HSV.
#
# A saturacao minima foi aumentada para reduzir
# a deteccao de tons de pele, cabelo e fundo.

limite1_baixo = np.array(
    [0, 150, 100]
)

limite1_alto = np.array(
    [10, 255, 255]
)

limite2_baixo = np.array(
    [170, 150, 100]
)

limite2_alto = np.array(
    [180, 255, 255]
)

mascara1 = cv2.inRange(
    hsv,
    limite1_baixo,
    limite1_alto
)

mascara2 = cv2.inRange(
    hsv,
    limite2_baixo,
    limite2_alto
)

mascara = cv2.bitwise_or(
    mascara1,
    mascara2
)


# ==========================================================
# OPERACOES MORFOLOGICAS
# ==========================================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

# Remove pequenos pontos isolados.
mascara = cv2.morphologyEx(
    mascara,
    cv2.MORPH_OPEN,
    kernel
)

# Fecha pequenos espacos dentro
# da regiao segmentada.
mascara = cv2.morphologyEx(
    mascara,
    cv2.MORPH_CLOSE,
    kernel
)


# ==========================================================
# ENCONTRAR CONTORNOS
# ==========================================================

contornos, _ = cv2.findContours(
    mascara,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

roi = None
caixa_roi = None

altura_frame, largura_frame = (
    frame_pipeline.shape[:2]
)

area_frame = (
    altura_frame
    * largura_frame
)


# ==========================================================
# FILTRAR CONTORNOS
# ==========================================================

contornos_validos = []

for contorno in contornos:

    area = cv2.contourArea(
        contorno
    )

    x, y, w, h = cv2.boundingRect(
        contorno
    )

    area_caixa = (
        w * h
    )

    # Ignora pequenos ruidos.
    if area < 1000:
        continue

    # Evita regioes que ocupem uma
    # parte muito grande do frame.
    if area_caixa > area_frame * 0.15:
        continue

    # Evita regioes excessivamente largas.
    if w > largura_frame * 0.60:
        continue

    # Evita regioes excessivamente altas.
    if h > altura_frame * 0.60:
        continue

    # Calcula a proporcao da caixa.
    proporcao = (
        w / float(h)
    )

    # A ROI procurada deve possuir
    # proporcoes compativeis com um objeto.
    if proporcao < 0.40:
        continue

    if proporcao > 2.00:
        continue

    contornos_validos.append(
        contorno
    )


# ==========================================================
# SELECIONAR ROI
# ==========================================================

if contornos_validos:

    maior = max(
        contornos_validos,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        maior
    )

    caixa_roi = (
        x,
        y,
        w,
        h
    )

    # Adiciona uma pequena margem ao redor
    # do objeto para fornecer contexto a DNN.
    margem = 15

    x1 = max(
        0,
        x - margem
    )

    y1 = max(
        0,
        y - margem
    )

    x2 = min(
        largura_frame,
        x + w + margem
    )

    y2 = min(
        altura_frame,
        y + h + margem
    )

    roi = frame_pipeline[
        y1:y2,
        x1:x2
    ]

fim = time.perf_counter()

tempo_hsv = (
    fim - inicio
) * 1000


# ==========================================================
# ETAPA 3 - FEATURES ORB
# ==========================================================

inicio = time.perf_counter()

orb = cv2.ORB_create(
    nfeatures=500
)

cinza = cv2.cvtColor(
    frame_pipeline,
    cv2.COLOR_BGR2GRAY
)

pontos, descritores = (
    orb.detectAndCompute(
        cinza,
        None
    )
)

fim = time.perf_counter()

tempo_orb = (
    fim - inicio
) * 1000


# ==========================================================
# ETAPA 4 - HAAR CASCADE
# ==========================================================

inicio = time.perf_counter()

arquivo_haar = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(
    arquivo_haar
)

faces = detector.detectMultiScale(
    cinza,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

fim = time.perf_counter()

tempo_haar = (
    fim - inicio
) * 1000


# ==========================================================
# ETAPA 5 - CLASSIFICACAO DNN
# ==========================================================

inicio = time.perf_counter()

classe_dnn = None
confianca_dnn = 0.0

if roi is not None and roi.size > 0:

    prob = predict_opencv(
        roi,
        net
    )

    resultado_dnn = topk(
        prob,
        1
    )[0]

    indice = resultado_dnn[0]
    confianca_dnn = resultado_dnn[1]

    classe_dnn = labels[
        indice
    ]

fim = time.perf_counter()

tempo_dnn = (
    fim - inicio
) * 1000


# ==========================================================
# FRAME FINAL
# ==========================================================

resultado = frame_pipeline.copy()


# ==========================================================
# DESENHAR ROI HSV
# ==========================================================

if caixa_roi is not None:

    x, y, w, h = caixa_roi

    cv2.rectangle(
        resultado,
        (x, y),
        (x+w, y+h),
        (0, 255, 255),
        2
    )

    posicao_texto_x = max(
        10,
        x
    )

    posicao_texto_y = max(
        25,
        y - 10
    )

    cv2.putText(
        resultado,
        "ROI HSV",
        (
            posicao_texto_x,
            posicao_texto_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# ==========================================================
# DESENHAR FEATURES ORB
# ==========================================================

resultado = cv2.drawKeypoints(
    resultado,
    pontos,
    None,
    color=(255, 0, 0),
    flags=0
)


# ==========================================================
# DESENHAR DETECCOES HAAR
# ==========================================================

for x, y, w, h in faces:

    cv2.rectangle(
        resultado,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )

    posicao_haar_x = max(
        10,
        x
    )

    posicao_haar_y = max(
        25,
        y - 10
    )

    cv2.putText(
        resultado,
        "Haar",
        (
            posicao_haar_x,
            posicao_haar_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# ==========================================================
# DESENHAR CLASSIFICACAO DNN
# ==========================================================

if (
    classe_dnn is not None
    and caixa_roi is not None
):

    x, y, w, h = caixa_roi

    texto = (
        f"DNN: {classe_dnn} "
        f"{confianca_dnn * 100:.1f}%"
    )

    posicao_dnn_x = max(
        10,
        x
    )

    posicao_dnn_y = min(
        resultado.shape[0] - 15,
        y + h + 25
    )

    cv2.putText(
        resultado,
        texto,
        (
            posicao_dnn_x,
            posicao_dnn_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


# ==========================================================
# TEMPO TOTAL
# ==========================================================

tempo_total = (
    tempo_undistort
    + tempo_hsv
    + tempo_orb
    + tempo_haar
    + tempo_dnn
)


# ==========================================================
# INFORMACOES NO FRAME
# ==========================================================

cv2.putText(
    resultado,
    f"ORB: {len(pontos)} pontos",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    resultado,
    f"Haar: {len(faces)} deteccoes",
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)


# ==========================================================
# EXIBIR TEMPOS
# ==========================================================

print(
    "\n=== TEMPO DO PIPELINE ==="
)

print(
    f"1 - Undistort: "
    f"{tempo_undistort:.2f} ms"
)

print(
    f"2 - Segmentacao HSV: "
    f"{tempo_hsv:.2f} ms"
)

print(
    f"3 - ORB: "
    f"{tempo_orb:.2f} ms"
)

print(
    f"4 - Haar Cascade: "
    f"{tempo_haar:.2f} ms"
)

print(
    f"5 - Classificacao DNN: "
    f"{tempo_dnn:.2f} ms"
)

print(
    "----------------------------"
)

print(
    f"Tempo total: "
    f"{tempo_total:.2f} ms"
)


# ==========================================================
# RESULTADOS
# ==========================================================

print(
    "\n=== RESULTADOS ==="
)

print(
    "Features ORB:",
    len(pontos)
)

print(
    "Deteccoes Haar:",
    len(faces)
)

if caixa_roi is not None:

    print(
        "ROI HSV:",
        "detectada"
    )

    print(
        "Posicao ROI:",
        caixa_roi
    )

else:

    print(
        "ROI HSV:",
        "nao detectada"
    )

if classe_dnn is not None:

    print(
        "Classe DNN:",
        classe_dnn
    )

    print(
        "Confianca:",
        f"{confianca_dnn * 100:.2f}%"
    )

else:

    print(
        "Nenhuma ROI HSV encontrada "
        "para classificacao."
    )


# ==========================================================
# SALVAR FRAME FINAL
# ==========================================================

cv2.imwrite(
    str(arquivo_saida),
    resultado
)

print(
    "\nFrame final salvo em:",
    arquivo_saida
)

print(
    "Resultado do undistort salvo em:",
    arquivo_undistort
)


# ==========================================================
# EXIBIR FRAME FINAL
# ==========================================================

# Reduz somente a imagem exibida na tela.
# O arquivo salvo permanece na resolucao original.

escala = 0.5

frame_exibicao = cv2.resize(
    resultado,
    None,
    fx=escala,
    fy=escala,
    interpolation=cv2.INTER_AREA
)

cv2.imshow(
    "Pipeline Completo",
    frame_exibicao
)

cv2.waitKey(0)
cv2.destroyAllWindows()