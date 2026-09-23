"""
EXERCÍCIO 1 - ITEM A
Calibração de câmera

Neste exercício são utilizadas imagens sintéticas de um tabuleiro de
xadrez com 7 x 6 cantos internos. O conjunto contém diferentes posições,
inclinações e distâncias, simulando as capturas necessárias para realizar
a calibração de uma câmera.

O pipeline realiza:
1. geração de pelo menos 15 imagens;
2. detecção e refinamento dos cantos;
3. calibração da câmera;
4. cálculo dos erros de reprojeção;
5. armazenamento dos parâmetros da calibração;
6. correção da distorção;
7. comparação entre imagem original e corrigida.
"""

from pathlib import Path

import cv2
import numpy as np

from synthetic_utils import (
    generate_dataset,
    calibrate_from_paths,
    reprojection_errors
)


ROOT = Path(__file__).resolve().parent


# ==========================================================
# ETAPA 1 - Gerar as imagens do tabuleiro
# ==========================================================

# O enunciado solicita pelo menos 15 imagens.
# São utilizadas 18 imagens com ângulos e distâncias variados.

paths = generate_dataset(
    ROOT,
    n=18
)

print("=== IMAGENS GERADAS ===")
print(f"Total de imagens: {len(paths)}")

for path in paths:
    print(" -", path.name)


# ==========================================================
# ETAPA 2 - Calibração da câmera
# ==========================================================

(
    rms,
    K,
    dist,
    rvecs,
    tvecs,
    objpoints,
    imgpoints,
    used,
    image_size
) = calibrate_from_paths(paths)


# ==========================================================
# ETAPA 3 - Matriz intrínseca
# ==========================================================

print("\n=== MATRIZ INTRÍNSECA K ===")
print(K)

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

print("\nParâmetros intrínsecos:")
print(f"fx = {fx:.4f} px")
print(f"fy = {fy:.4f} px")
print(f"cx = {cx:.4f} px")
print(f"cy = {cy:.4f} px")

# Significado físico da matriz K:
#
# fx e fy representam as distâncias focais da câmera
# medidas em pixels.
#
# cx e cy representam as coordenadas do ponto principal,
# isto é, o ponto onde o eixo óptico da câmera encontra
# o plano da imagem.


# ==========================================================
# ETAPA 4 - Coeficientes de distorção
# ==========================================================

d = dist.ravel()

k1, k2, p1, p2, k3 = d[:5]

print("\n=== COEFICIENTES DE DISTORÇÃO ===")

print(f"k1 = {k1:.8f}")
print(f"k2 = {k2:.8f}")
print(f"p1 = {p1:.8f}")
print(f"p2 = {p2:.8f}")
print(f"k3 = {k3:.8f}")

# Significado físico:
#
# k1, k2 e k3 representam a distorção radial da lente.
# Essa distorção pode fazer linhas retas parecerem curvas,
# principalmente nas regiões próximas às bordas da imagem.
#
# p1 e p2 representam a distorção tangencial.
# Ela ocorre quando a lente e o sensor não estão
# perfeitamente alinhados.


# ==========================================================
# ETAPA 5 - Erro de reprojeção
# ==========================================================

errors = reprojection_errors(
    K,
    dist,
    rvecs,
    tvecs,
    objpoints,
    imgpoints
)

print("\n=== ERRO DE REPROJEÇÃO POR IMAGEM ===")

for path, erro in zip(used, errors):
    print(
        f"{path.name}: {erro:.4f} px"
    )

mean_error = np.mean(errors)

print(
    f"\nErro médio de reprojeção: "
    f"{mean_error:.4f} px"
)

print(
    f"RMS retornado pelo calibrateCamera: "
    f"{rms:.4f}"
)

# Referência prática para o erro de reprojeção:
#
# 0 a 0,5 px:
# geralmente representa uma calibração muito boa.
#
# 0,5 a 1,0 px:
# frequentemente representa uma calibração utilizável.
#
# Acima de 1 px:
# convém investigar a qualidade das imagens,
# a detecção dos cantos e a variedade das poses.
#
# Para aplicações robóticas, quanto menor o erro,
# maior tende a ser a precisão das estimativas.
# Entretanto, o limite aceitável depende da aplicação.


# ==========================================================
# ETAPA 6 - Salvar parâmetros da calibração
# ==========================================================

arquivo_calibracao = (
    ROOT
    / "saidas"
    / "calibracao_camera.npz"
)

arquivo_calibracao.parent.mkdir(
    exist_ok=True
)

np.savez(
    arquivo_calibracao,
    K=K,
    dist=dist
)

print(
    "\nCalibração salva em:",
    arquivo_calibracao
)


# ==========================================================
# ETAPA 7 - Correção da distorção
# ==========================================================

img = cv2.imread(
    str(used[0])
)

if img is None:
    raise RuntimeError(
        "Não foi possível carregar a imagem."
    )

corrigida = cv2.undistort(
    img,
    K,
    dist
)


# ==========================================================
# ETAPA 8 - Painel original e corrigida
# ==========================================================

painel = np.hstack([
    img,
    corrigida
])

cv2.putText(
    painel,
    "ORIGINAL",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 0, 255),
    3
)

cv2.putText(
    painel,
    "CORRIGIDA",
    (img.shape[1] + 30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 120, 0),
    3
)

saida = (
    ROOT
    / "saidas"
    / "painel_original_corrigida.png"
)

saida.parent.mkdir(
    exist_ok=True
)

# Salva o painel na resolução original.
cv2.imwrite(
    str(saida),
    painel
)

print(
    "\nPainel salvo em:",
    saida
)

# Reduz somente para exibição na tela.
escala = 0.5

painel_exibicao = cv2.resize(
    painel,
    None,
    fx=escala,
    fy=escala,
    interpolation=cv2.INTER_AREA
)

cv2.imshow(
    "Original | Corrigida",
    painel_exibicao
)

cv2.waitKey(0)
cv2.destroyAllWindows()