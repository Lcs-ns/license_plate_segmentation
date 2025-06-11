import pytesseract
from PIL import Image
import os
import csv
import shutil

# === CONFIGURAÇÕES ===
CAMINHO_IMAGENS = "caracteres_segmentados"
ARQUIVO_SAIDA = os.path.join(CAMINHO_IMAGENS, "rotulos.csv")
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Garante que só arquivos char_*.png sejam processados
arquivos = sorted(
    [f for f in os.listdir(CAMINHO_IMAGENS) if f.startswith("char_") and f.endswith(".png")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Processa e move os arquivos
with open(ARQUIVO_SAIDA, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["arquivo", "rotulo"])  # Cabeçalho CSV

    for arquivo in arquivos:
        caminho_imagem = os.path.join(CAMINHO_IMAGENS, arquivo)
        imagem = Image.open(caminho_imagem)

        # OCR com Tesseract para um único caractere
        rotulo = pytesseract.image_to_string(
            imagem,
            config=f'--psm 10 -c tessedit_char_whitelist={WHITELIST}'
        ).strip().upper()

        # Verificação mínima de validade
        if rotulo not in WHITELIST or len(rotulo) != 1:
            print(f"(!) Ignorado: {arquivo} -> rótulo inválido '{rotulo}'")
            continue

        # Cria pasta do rótulo se necessário
        pasta_destino = os.path.join(CAMINHO_IMAGENS, rotulo)
        os.makedirs(pasta_destino, exist_ok=True)

        # Move a imagem para a pasta correta
        novo_caminho = os.path.join(pasta_destino, arquivo)
        shutil.move(caminho_imagem, novo_caminho)

        print(f"{arquivo} -> {rotulo}")
        writer.writerow([arquivo, rotulo])
