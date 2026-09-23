"""
Exercicio 2 - Item A
Medicao aproximada de memoria do Keras.
"""

import os
import psutil
import cv2

from tensorflow.keras.applications import (
    MobileNetV2
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

from dnn_utils import (
    list_classification_images
)


processo = psutil.Process(
    os.getpid()
)

memoria_antes = (
    processo.memory_info().rss
    / (1024 * 1024)
)

modelo = MobileNetV2(
    weights="imagenet"
)

img = cv2.imread(
    str(
        list_classification_images()[0]
    )
)

rgb = cv2.cvtColor(
    cv2.resize(
        img,
        (224, 224)
    ),
    cv2.COLOR_BGR2RGB
)

entrada = preprocess_input(
    rgb.astype("float32")
)[None, ...]

_ = modelo.predict(
    entrada,
    verbose=0
)

memoria_depois = (
    processo.memory_info().rss
    / (1024 * 1024)
)

aumento = (
    memoria_depois
    - memoria_antes
)

print(
    f"Memoria antes: "
    f"{memoria_antes:.1f} MB"
)

print(
    f"Memoria depois: "
    f"{memoria_depois:.1f} MB"
)

print(
    f"Aumento aproximado: "
    f"{aumento:.1f} MB"
)