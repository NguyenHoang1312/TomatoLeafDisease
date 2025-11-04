import json
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== CONST ====
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 64
EPOCHS = 15

TRAIN_DIR = 'tomato/train'
VAL_DIR = 'tomato/val'
RESULT_DIR = 'results'

# ==== Tạo class tăng cường dữ liệu cho ảnh ====
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# ==== Lưu class_indices để tra ngược khi dự đoán ====
class_indices = train_data.class_indices
with open(RESULT_DIR + '/class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Đã lưu class_indices.json")

# ==== Early stopping ====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==== Mô hình CNN ====
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.BatchNormalization(),
    layers.Dense(len(class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==== Huấn luyện ====
start_time = time.time()

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop]
)

end_time = time.time()

# ==== Lưu mô hình ====
model.save(RESULT_DIR + '/tomato_leaf_disease_model.keras')
print('Đã lưu mô hình vào tomato_leaf_disease_model.keras')

# ==== Thời gian huấn luyện ====
training_time = end_time - start_time
print(f'Trained time: {training_time/60 :.2f} minutes')

# ==== Biểu đồ loss & accuracy ====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linestyle='-', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--', marker='x')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle=':', marker='s')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linestyle='-.', marker='^')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(RESULT_DIR + "/training_plot.png")
plt.show()
plt.close()
print("Đã lưu training_plot.png")
