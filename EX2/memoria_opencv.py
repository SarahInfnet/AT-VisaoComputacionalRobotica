"""
Exercicio 2 - Item A
Medicao aproximada de memoria do OpenCV DNN.
"""

import os
import psutil
import cv2

from dnn_utils import (
    list_classification_images,
    get_opencv_net,
    predict_opencv
)


processo = psutil.Process(
    os.getpid()
)

memoria_antes = (
    processo.memory_info().rss
    / (1024 * 1024)
)

net = get_opencv_net()

img = cv2.imread(
    str(
        list_classification_images()[0]
    )
)

_ = predict_opencv(
    img,
    net
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