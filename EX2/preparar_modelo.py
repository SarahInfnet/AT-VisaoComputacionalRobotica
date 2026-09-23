"""
Exercicio 2 - Item A
Preparacao da MobileNetV2 para uso no OpenCV DNN.

O modelo MobileNetV2 pre-treinado no ImageNet e carregado
pelo Keras e convertido para TensorFlow Lite.
Tambem sao baixadas as 1000 classes do ImageNet.
"""

from pathlib import Path
import urllib.request
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

from dnn_utils import MODELS, ensure_dirs


# ==========================================================
# PREPARAR PASTAS
# ==========================================================

ensure_dirs()


# ==========================================================
# BAIXAR LABELS DO IMAGENET
# ==========================================================

arquivo_labels = MODELS / "imagenet_labels.txt"

if not arquivo_labels.exists():

    url = (
        "https://raw.githubusercontent.com/"
        "pytorch/hub/master/imagenet_classes.txt"
    )

    print("Baixando labels ImageNet...")

    urllib.request.urlretrieve(
        url,
        arquivo_labels
    )

    print(
        "Labels salvas em:",
        arquivo_labels
    )

else:

    print(
        "Labels ja existem:",
        arquivo_labels
    )


# ==========================================================
# CONVERTER MOBILENETV2 PARA TFLITE
# ==========================================================

arquivo_modelo = (
    MODELS
    / "mobilenetv2_imagenet.tflite"
)

if not arquivo_modelo.exists():

    print("\nCarregando MobileNetV2...")

    modelo = MobileNetV2(
        weights="imagenet"
    )

    print("Convertendo para TensorFlow Lite...")

    conversor = (
        tf.lite.TFLiteConverter
        .from_keras_model(modelo)
    )

    conversor.optimizations = []

    modelo_tflite = conversor.convert()

    arquivo_modelo.write_bytes(
        modelo_tflite
    )

    print(
        "Modelo salvo em:",
        arquivo_modelo
    )

    print(
        "Tamanho aproximado [MB]:",
        arquivo_modelo.stat().st_size
        / (1024 * 1024)
    )

else:

    print(
        "Modelo ja existe:",
        arquivo_modelo
    )


print("\nPreparacao concluida.")