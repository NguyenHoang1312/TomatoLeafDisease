import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==== Cấu hình ====
TEST_DIR = 'tomato/test/Tomato___Early_blight'
MODEL_PATH = 'results/tomato_leaf_disease_model_mobilenetv2.keras'
CLASS_INDICES_PATH = 'results/class_indices.json'
IMAGE_SIZE = (256, 256)

# ==== Tải mô hình và class indices ====
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_label = {v: k for k, v in class_indices.items()}

# ==== Lấy danh sách file ảnh ====
image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise FileNotFoundError(f'Không tìm thấy ảnh trong thư mục {TEST_DIR}')

# ==== Tiền xử lý ảnh ====
images_rgb = []
image_paths = []
img_batch = []

for file_name in image_files:
    path = os.path.join(TEST_DIR, file_name)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        continue

    img_resized = cv2.resize(img_bgr, IMAGE_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0

    images_rgb.append(img_rgb)
    image_paths.append(file_name)
    img_batch.append(img_normalized)

# ==== Dự đoán ====
img_batch = np.array(img_batch)
preds = model.predict(img_batch)

# ==== Hiển thị kết quả ====
n = len(images_rgb)
cols = 4
rows = (n + cols - 1) // cols

plt.figure(figsize=(4 * cols, 4 * rows))
for i in range(n):
    pred_class_index = int(np.argmax(preds[i]))
    pred_class_label = idx_to_label[pred_class_index]
    confidence = preds[i][pred_class_index]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(images_rgb[i])
    plt.title(f'{pred_class_label}\n({confidence * 100:.1f}%)', fontsize=20)
    plt.axis("off")

plt.tight_layout()
plt.show()

