import cv2
import os
import easyocr

reader = easyocr.Reader(['pt','en'], gpu=False)

input_folder  = "placas_recortadas"
output_folder = "caracteres_por_label"
standard_size = (200, 60)

# Parâmetros fixos de filtro
min_h = standard_size[1] * 0.3   # 18 px
min_w = standard_size[0] * 0.02  # 4 px

# Cria pasta raiz
os.makedirs(output_folder, exist_ok=True)

# Arquivos de placa
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
])

for idx, filename in enumerate(image_files, start=1):
    # 1) Carrega e redimensiona
    img = cv2.imread(os.path.join(input_folder, filename))
    if img is None:
        print(f"[Aviso] Não foi possível ler {filename}, pulando.")
        continue
    plate = cv2.resize(img, standard_size)

    # 2) Binariza
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    )

    # 3) Segmenta contornos e ordena
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(c) for c in contours]
    paired = sorted(zip(boxes, contours), key=lambda b: b[0][0])
    sorted_contours = [c for (_, c) in paired]

    # 4) Para cada caractere, faz OCR e salva
    for c_idx, cnt in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_h or w < min_w or not (1.0 < h/w < 7.0):
            continue

        char_img = thresh[y:y+h, x:x+w]

        # OCR de caractere único
        result = reader.readtext(char_img, 
                         allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                         detail=0,       # só texto puro
                         paragraph=False # um bloco por vez
                        )
        label = result[0] if result else 'unknown'
        if len(label) != 1:
            label = "unknown"

        # Cria pasta do label
        label_folder = os.path.join(output_folder, label)
        os.makedirs(label_folder, exist_ok=True)

        # Salva com nome único
        out_name = f"{os.path.splitext(filename)[0]}_{c_idx}.png"
        cv2.imwrite(os.path.join(label_folder, out_name), char_img)

    print(f"[{idx}/{len(image_files)}] {filename} → character folders updated.")
