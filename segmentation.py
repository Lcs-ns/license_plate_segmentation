import os
import cv2

# Parâmetros e pastas
input_folder   = 'placas_recortadas'
output_folder  = 'caracteres_segmentados'
standard_size  = (200, 60)   # largura x altura padronizadas
min_h_ratio    = 0.3         # 30% da altura
min_w_ratio    = 0.02        #  2% da largura
aspect_min     = 1.0
aspect_max     = 7.0

os.makedirs(output_folder, exist_ok=True)
img_counter = 0

# Lista de imagens
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for idx, filename in enumerate(image_files, start=1):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] não foi possível abrir {filename}")
        continue

    # 1) Redimensiona para tamanho fixo
    plate = cv2.resize(img, standard_size)

    # 2) Converte para cinza
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 3) Binarização com Otsu (inverte para ter caracteres brancos)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4) Remove pequenos ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5) Encontra contornos externos
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 6) Ordena contornos da esquerda para a direita
    boxes = [cv2.boundingRect(c) for c in contours]
    paired = sorted(zip(boxes, contours), key=lambda bc: bc[0][0])
    sorted_contours = [c for (_, c) in paired]

    h_img, w_img = clean.shape
    min_h = h_img * min_h_ratio
    min_w = w_img * min_w_ratio

    saved = 0
    # 7) Para cada contorno válido, salva o crop tratado
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # aplica filtros fixos
        if h < min_h or w < min_w:
            continue
        ratio = h / w
        if not (aspect_min < ratio < aspect_max):
            continue

        # recorta no clean (binário) ou no gray, conforme preferir
        char_bin = clean[y:y+h, x:x+w]
        # opcional: voltar ao formato BGR para salvar colorido
        # char_color = plate[y:y+h, x:x+w]

        out_path = os.path.join(output_folder, f'char_{img_counter}.png')
        cv2.imwrite(out_path, char_bin)
        img_counter += 1
        saved += 1

    print(
        f"[{idx}/{len(image_files)}] "
        f"{filename} → {saved} caracteres extraídos."
    )

print(f"\nTotal de crops salvos: {img_counter}")
