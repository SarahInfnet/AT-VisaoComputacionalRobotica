"""
Exercicio 2 - Item A
Classificacao com OpenCV DNN e comparacao com Keras.

Etapas:
1. Carregar MobileNetV2 no OpenCV DNN e no Keras.
2. Processar 10 imagens de categorias distintas.
3. Obter Top-3 e salvar imagens anotadas.
4. Comparar resultados do OpenCV DNN com Keras.
5. Medir latencia de inferencia.
6. Calcular acuracia Top-1.
7. Exibir resultados no terminal.

A memoria e medida separadamente nos arquivos:
- memoria_opencv.py
- memoria_keras.py
"""

import csv
import cv2
import pandas as pd

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from dnn_utils import (
    ROOT,
    DATA_CLASS,
    OUT,
    ensure_dirs,
    list_classification_images,
    get_opencv_net,
    predict_opencv,
    topk,
    draw_top3,
    load_labels,
    time_ms
)


# ==========================================================
# PREPARACAO
# ==========================================================

ensure_dirs()

paths = list_classification_images()[:10]

if len(paths) < 10:
    raise RuntimeError(
        "Sao necessarias pelo menos 10 imagens."
    )

print("=== IMAGENS ENCONTRADAS ===")
print("Total:", len(paths))

for p in paths:
    print("-", p.name)


# ==========================================================
# CARREGAR LABELS DO IMAGENET
# ==========================================================

labels = load_labels()

print("\n=== LABELS IMAGENET ===")
print("Total de classes:", len(labels))


# ==========================================================
# CARREGAR MODELO OPENCV DNN
# ==========================================================

print("\n=== CARREGANDO OPENCV DNN ===")

net = get_opencv_net()

print("Modelo OpenCV DNN carregado.")
print(
    "Quantidade de camadas:",
    len(net.getLayerNames())
)


# ==========================================================
# CARREGAR MODELO KERAS
# ==========================================================

print("\n=== CARREGANDO KERAS ===")

modelo_keras = MobileNetV2(
    weights="imagenet"
)

print("Modelo Keras carregado.")
print(
    "Entrada:",
    modelo_keras.input_shape
)
print(
    "Saida:",
    modelo_keras.output_shape
)


# ==========================================================
# PREPARAR IMAGEM PARA KERAS
# ==========================================================

def preparar_keras(img):

    rgb = cv2.cvtColor(
        cv2.resize(
            img,
            (224, 224)
        ),
        cv2.COLOR_BGR2RGB
    )

    entrada = preprocess_input(
        rgb.astype("float32")
    )

    return entrada[None, ...]


# ==========================================================
# TOP-3 COM OPENCV DNN
# ==========================================================

print(
    "\n=== TOP-3 - OPENCV DNN ==="
)

for p in paths:

    img = cv2.imread(
        str(p)
    )

    if img is None:
        print(
            "Nao foi possivel abrir:",
            p.name
        )
        continue

    prob = predict_opencv(
        img,
        net
    )

    top3 = topk(
        prob,
        3
    )

    print(
        f"\nImagem: {p.name}"
    )

    for pos, (idx, conf) in enumerate(
        top3,
        start=1
    ):

        print(
            f"{pos}) "
            f"{labels[idx]}: "
            f"{conf * 100:.2f}%"
        )

    vis = draw_top3(
        img,
        top3,
        labels
    )

    arquivo_saida = (
        OUT
        / f"top3_{p.stem}.jpg"
    )

    cv2.imwrite(
        str(arquivo_saida),
        vis
    )


print(
    "\nImagens com Top-3 salvas em:",
    OUT
)


# ==========================================================
# COMPARAR OPENCV DNN X KERAS
# ==========================================================

print(
    "\n=== COMPARACAO TOP-3 ==="
)

for p in paths:

    img = cv2.imread(
        str(p)
    )

    if img is None:
        continue

    # ------------------------------
    # OpenCV DNN
    # ------------------------------

    prob_opencv = predict_opencv(
        img,
        net
    )

    top3_opencv = topk(
        prob_opencv,
        3
    )

    # ------------------------------
    # Keras
    # ------------------------------

    entrada_keras = preparar_keras(
        img
    )

    prob_keras = modelo_keras.predict(
        entrada_keras,
        verbose=0
    )[0]

    top3_keras = topk(
        prob_keras,
        3
    )

    # ------------------------------
    # Exibir resultados
    # ------------------------------

    print(
        f"\nImagem: {p.name}"
    )

    print("OpenCV DNN:")

    for pos, (idx, conf) in enumerate(
        top3_opencv,
        start=1
    ):

        print(
            f"  {pos}) "
            f"{labels[idx]} "
            f"{conf * 100:.2f}%"
        )

    print("Keras:")

    for pos, (idx, conf) in enumerate(
        top3_keras,
        start=1
    ):

        print(
            f"  {pos}) "
            f"{labels[idx]} "
            f"{conf * 100:.2f}%"
        )


# ==========================================================
# MEDIR LATENCIA
# ==========================================================

print(
    "\n=== LATENCIA ==="
)

imagem_teste = cv2.imread(
    str(paths[0])
)

entrada_keras = preparar_keras(
    imagem_teste
)

lat_opencv, desvio_opencv = time_ms(
    lambda: predict_opencv(
        imagem_teste,
        net
    ),
    loops=20,
    warmup=5
)

lat_keras, desvio_keras = time_ms(
    lambda: modelo_keras.predict(
        entrada_keras,
        verbose=0
    ),
    loops=20,
    warmup=5
)

print(
    f"OpenCV DNN: "
    f"{lat_opencv:.2f} "
    f"+/- {desvio_opencv:.2f} ms"
)

print(
    f"Keras: "
    f"{lat_keras:.2f} "
    f"+/- {desvio_keras:.2f} ms"
)


# ==========================================================
# CALCULAR ACURACIA TOP-1
# ==========================================================

print(
    "\n=== ACURACIA TOP-1 ==="
)

csv_path = (
    ROOT
    / "data"
    / "labels_top1.csv"
)

if not csv_path.exists():
    raise FileNotFoundError(
        "Arquivo labels_top1.csv "
        "nao encontrado."
    )


corretos_opencv = 0
corretos_keras = 0
total = 0


with open(
    csv_path,
    encoding="utf-8"
) as arquivo:

    reader = csv.DictReader(
        arquivo
    )

    for row in reader:

        esperado = (
            row["label_esperado"]
            .strip()
            .lower()
        )

        if not esperado:
            continue

        img_path = (
            DATA_CLASS
            / row["arquivo"]
        )

        img = cv2.imread(
            str(img_path)
        )

        if img is None:
            print(
                "Nao foi possivel abrir:",
                row["arquivo"]
            )
            continue

        # ------------------------------
        # OpenCV DNN
        # ------------------------------

        prob_opencv = predict_opencv(
            img,
            net
        )

        idx_opencv = topk(
            prob_opencv,
            1
        )[0][0]

        pred_opencv = (
            labels[idx_opencv]
            .lower()
            .replace(" ", "_")
        )

        acertou_opencv = (
            esperado in pred_opencv
            or pred_opencv in esperado
        )

        # ------------------------------
        # Keras
        # ------------------------------

        entrada = preparar_keras(
            img
        )

        prob_keras = modelo_keras.predict(
            entrada,
            verbose=0
        )[0]

        idx_keras = topk(
            prob_keras,
            1
        )[0][0]

        pred_keras = (
            labels[idx_keras]
            .lower()
            .replace(" ", "_")
        )

        acertou_keras = (
            esperado in pred_keras
            or pred_keras in esperado
        )

        # ------------------------------
        # Contabilizar resultados
        # ------------------------------

        corretos_opencv += int(
            acertou_opencv
        )

        corretos_keras += int(
            acertou_keras
        )

        total += 1

        print(
            f"\n{row['arquivo']}"
        )

        print(
            "Esperado:",
            esperado
        )

        print(
            "OpenCV:",
            pred_opencv,
            "| correto:",
            acertou_opencv
        )

        print(
            "Keras:",
            pred_keras,
            "| correto:",
            acertou_keras
        )


# ==========================================================
# RESULTADO DA ACURACIA
# ==========================================================

acc_opencv = (
    corretos_opencv / total
    if total
    else 0.0
)

acc_keras = (
    corretos_keras / total
    if total
    else 0.0
)

print(
    f"\nOpenCV DNN: "
    f"{acc_opencv * 100:.1f}% "
    f"({corretos_opencv}/{total})"
)

print(
    f"Keras: "
    f"{acc_keras * 100:.1f}% "
    f"({corretos_keras}/{total})"
)


# ==========================================================
# TABELA PARCIAL
# ==========================================================

print(
    "\n=== TABELA COMPARATIVA ==="
)

dados = [
    {
        "backend": "OpenCV DNN",
        "latencia_ms": round(
            lat_opencv,
            2
        ),
        "top1_acc": round(
            acc_opencv * 100,
            1
        )
    },
    {
        "backend": "Keras",
        "latencia_ms": round(
            lat_keras,
            2
        ),
        "top1_acc": round(
            acc_keras * 100,
            1
        )
    }
]

df = pd.DataFrame(
    dados
)

print(
    df.to_string(
        index=False
    )
)


# ==========================================================
# COMENTARIO TECNICO
# ==========================================================

print(
    "\n=== COMENTARIO TECNICO ==="
)

print(
    "OpenCV DNN tende a ser preferivel "
    "quando o objetivo e inferencia leve, "
    "integracao direta com o pipeline OpenCV "
    "e menor dependencia de frameworks completos."
)

print(
    "Keras e mais adequado para treinamento, "
    "fine-tuning, experimentacao e validacao "
    "de modelos."
)


# ==========================================================
# SALVAR TABELA PARCIAL
# ==========================================================

arquivo_tabela = (
    OUT
    / "tabela_comparativa_parcial.csv"
)

df.to_csv(
    arquivo_tabela,
    index=False
)

print(
    "\nTabela parcial salva em:",
    arquivo_tabela
)

print(
    "\nOBSERVACAO:"
)

print(
    "O uso de memoria foi medido separadamente "
    "nos scripts memoria_opencv.py e memoria_keras.py."
)