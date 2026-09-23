"""
EXERCÍCIO 4 - ITEM A

Segmentação semântica com FCN-ResNet50 e comparação
com segmentação clássica por cor HSV.

O programa:
1. Processa 5 imagens externas.
2. Gera a máscara semântica colorida.
3. Cria overlay semitransparente.
4. Calcula a porcentagem de área por classe.
5. Realiza segmentação HSV da pista.
6. Compara HSV e segmentação semântica lado a lado.

Discussão:
A segmentação HSV utiliza limites de cor definidos manualmente.
É simples e rápida, mas depende bastante da iluminação e pode
confundir regiões com cores semelhantes.

A segmentação semântica utiliza um modelo profundo pré-treinado
para classificar os pixels de acordo com categorias aprendidas.
Ela consegue representar diferentes objetos da cena, mas exige
mais processamento e depende das classes presentes no treinamento.

Para veículos autônomos, a segmentação semântica fornece mais
informações sobre a cena, enquanto técnicas como HSV podem ser
úteis em tarefas específicas e mais simples.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import (
    fcn_resnet50,
    FCN_ResNet50_Weights
)


# ==========================================================
# CAMINHOS
# ==========================================================

ROOT = Path(__file__).resolve().parent

PASTA_DADOS = ROOT / "dados"
PASTA_SAIDAS = ROOT / "saidas"

PASTA_SAIDAS.mkdir(
    exist_ok=True
)

IMAGENS = [
    PASTA_DADOS / "cena_01.jpg",
    PASTA_DADOS / "cena_02.jpg",
    PASTA_DADOS / "cena_03.jpg",
    PASTA_DADOS / "cena_04.jpg",
    PASTA_DADOS / "cena_05.jpg"
]


# ==========================================================
# MODELO FCN-RESNET50
# ==========================================================

print("\n==========================================")
print("CARREGANDO FCN-RESNET50")
print("==========================================")

weights = FCN_ResNet50_Weights.DEFAULT

model = fcn_resnet50(
    weights=weights
).eval()

preprocess = weights.transforms()

classes = weights.meta.get(
    "categories",
    []
)

print("Modelo carregado: FCN-ResNet50")
print("Número de classes:", len(classes))


# ==========================================================
# PALETA DE CORES
# ==========================================================

rng = np.random.default_rng(1)

palette = rng.integers(
    0,
    255,
    (256, 3),
    dtype=np.uint8
)

# Fundo preto
palette[0] = [0, 0, 0]


# ==========================================================
# SEGMENTAÇÃO HSV
# ==========================================================

def hsv_pista(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [0, 0, 40]
    )

    upper = np.array(
        [179, 60, 150]
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # ------------------------------------------------------
    # Limpeza morfológica
    # ------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ==========================================================
# TÍTULO NO PAINEL
# ==========================================================

def adicionar_titulo(
    img,
    texto
):

    out = img.copy()

    cv2.rectangle(
        out,
        (0, 0),
        (out.shape[1], 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        out,
        texto,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return out


# ==========================================================
# PROCESSAMENTO
# ==========================================================

for caminho in IMAGENS:

    print("\n==========================================")
    print("IMAGEM:", caminho.name)
    print("==========================================")

    # ------------------------------------------------------
    # Verificação
    # ------------------------------------------------------

    if not caminho.exists():

        print(
            "Imagem não encontrada:",
            caminho
        )

        continue

    # ------------------------------------------------------
    # Imagem original
    # ------------------------------------------------------

    img = cv2.imread(
        str(caminho)
    )

    if img is None:

        print(
            "Erro ao carregar:",
            caminho
        )

        continue

    altura, largura = img.shape[:2]

    # ------------------------------------------------------
    # FCN-ResNet50
    # ------------------------------------------------------

    img_pil = Image.open(
        caminho
    ).convert("RGB")

    x = preprocess(
        img_pil
    ).unsqueeze(0)

    with torch.no_grad():

        out = model(
            x
        )["out"][0]

    mask = out.argmax(
        0
    ).byte().cpu().numpy()

    # ------------------------------------------------------
    # Máscara colorida
    # ------------------------------------------------------

    color = palette[
        mask
    ]

    color = cv2.resize(
        color,
        (largura, altura),
        interpolation=cv2.INTER_NEAREST
    )

    # ------------------------------------------------------
    # Overlay semântico
    # ------------------------------------------------------

    overlay_semantico = cv2.addWeighted(
        img,
        0.65,
        color,
        0.35,
        0
    )

    # ------------------------------------------------------
    # Porcentagem por classe
    # ------------------------------------------------------

    vals, counts = np.unique(
        mask,
        return_counts=True
    )

    print("\nPorcentagem por classe:")

    for v, c in zip(
        vals,
        counts
    ):

        pct = (
            100
            * c
            / mask.size
        )

        if pct > 0.01:

            if int(v) < len(classes):
                nome = classes[int(v)]
            else:
                nome = f"classe_{int(v)}"

            print(
                f"{nome:15s}: "
                f"{pct:5.2f}%"
            )

    # ------------------------------------------------------
    # HSV
    # ------------------------------------------------------

    mask_hsv = hsv_pista(
        img
    )

    hsv_color = np.zeros_like(
        img
    )

    hsv_color[
        mask_hsv > 0
    ] = (0, 255, 255)

    overlay_hsv = cv2.addWeighted(
        img,
        0.75,
        hsv_color,
        0.25,
        0
    )

    # ------------------------------------------------------
    # Painel comparativo
    # ------------------------------------------------------

    original_titulo = adicionar_titulo(
        img,
        "ORIGINAL"
    )

    hsv_titulo = adicionar_titulo(
        overlay_hsv,
        "HSV"
    )

    semantica_titulo = adicionar_titulo(
        overlay_semantico,
        "FCN-RESNET50"
    )

    painel = np.hstack(
        [
            original_titulo,
            hsv_titulo,
            semantica_titulo
        ]
    )

    # ------------------------------------------------------
    # Salvamento
    # ------------------------------------------------------

    nome = caminho.stem

    caminho_mascara = (
        PASTA_SAIDAS
        / f"{nome}_mascara_semantica.png"
    )

    caminho_overlay = (
        PASTA_SAIDAS
        / f"{nome}_overlay_semantico.png"
    )

    caminho_hsv = (
        PASTA_SAIDAS
        / f"{nome}_hsv.png"
    )

    caminho_painel = (
        PASTA_SAIDAS
        / f"{nome}_comparacao.png"
    )

    cv2.imwrite(
        str(caminho_mascara),
        color
    )

    cv2.imwrite(
        str(caminho_overlay),
        overlay_semantico
    )

    cv2.imwrite(
        str(caminho_hsv),
        overlay_hsv
    )

    cv2.imwrite(
        str(caminho_painel),
        painel
    )

    print("\nArquivos gerados:")
    print(caminho_mascara)
    print(caminho_overlay)
    print(caminho_hsv)
    print(caminho_painel)

    # ------------------------------------------------------
    # Exibição
    # ------------------------------------------------------

    painel_exibicao = cv2.resize(
        painel,
        None,
        fx=0.40,
        fy=0.40
    )

    cv2.imshow(
        "Original x HSV x FCN",
        painel_exibicao
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


print("\nProcessamento finalizado.")