import os
import json
import time
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ==== CONST ====
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 15

RESULT_DIR = 'results'
TRAIN_DIR = 'tomato/train'
VAL_DIR = 'tomato/val'

# ==== Tạo class tăng cường dữ liệu cho ảnh ====
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(
    rescale=1./255
)

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

# ==== Lưu class_indices ====
class_indices = train_data.class_indices
os.makedirs(RESULT_DIR, exist_ok=True)
with open(os.path.join(RESULT_DIR, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)
print("Đã lưu class_indices.json")

# ==== EarlyStopping ====
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==== Mô hình MobileNetV2 ====
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False

# ==== Cấu hình lớp đầu ra ====
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(len(class_indices), activation='softmax')(x)

# ==== Mô hình ====
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==== Huấn luyện lần 1 (feature extractor) ====
start_time_1 = time.time()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

end_time_1 = time.time()
print(f"Huấn luyện lần 1: {(end_time_1 - start_time_1)/60:.2f} minutes")

# ==== Fine-tune thêm (mở khóa 20 layer cuối) ====
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-5),  # tốc độ học nhỏ hơn
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==== Huấn luyện lần 2 (fine-tuning) ====
start_time_2 = time.time()

fine_tune_epochs = 5
history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=fine_tune_epochs,
    callbacks=[early_stop]
)

end_time_2 = time.time()

print(f"Huấn luyện lần 2 (fine-tune): {(end_time_2 - start_time_2)/60:.2f} minutes")

# ==== Lưu mô hình ====
model.save(os.path.join(RESULT_DIR, 'tomato_leaf_disease_model_mobilenetv2.keras'))
print("Đã lưu mô hình vaào tomato_leaf_disease_model_mobilenetv2.keras")

# ==== Biểu đồ loss & accuracy ====
def plot_history(h1, h2):
    plt.figure(figsize=(12, 5))

    # ==== Loss ====
    plt.subplot(1, 2, 1)

    # Trước fine-tuning (1 - 15)
    plt.plot(h1.history['loss'], label='Train Loss')
    plt.plot(h1.history['val_loss'], label='Val Loss')

    # Sau fine-tuning (16 - 20)
    plt.plot(
        [*range(len(h1.history['loss']), len(h1.history['loss']) + len(h2.history['loss']))],
        h2.history['loss'],
        label='Fine-tune Train Loss',
        linestyle='--'
    )
    plt.plot(
        [*range(len(h1.history['loss']), len(h1.history['val_loss']) + len(h2.history['val_loss']))],
        h2.history['val_loss'],
        label='Fine-tune Val Loss',
        linestyle='--'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # ==== Accuracy ====
    plt.subplot(1, 2, 2)

    # Trước fine-tuning (1 - 15)
    plt.plot(h1.history['accuracy'], label='Train Accuracy')
    plt.plot(h1.history['val_accuracy'], label='Val Accuracy')

    # Sau fine-tuning (16 - 20)
    plt.plot(
        [*range(len(h1.history['accuracy']), len(h1.history['accuracy']) + len(h2.history['accuracy']))],
        h2.history['accuracy'],
        label='Fine-tune Train Accuracy',
        linestyle='--'
    )
    plt.plot(
        [*range(len(h1.history['accuracy']), len(h1.history['val_accuracy']) + len(h2.history['val_accuracy']))],
        h2.history['val_accuracy'],
        label='Fine-tune Val Accuracy',
        linestyle='--'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()

    plt.savefig(os.path.join(RESULT_DIR, 'training_plot_mobilenetv2.png'))
    print("Saved training_plot_mobilenetv2.png")
plot_history(history, history_fine)