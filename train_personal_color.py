import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# 📁 경로 설정
DATA_DIR = 'data/personal_color'
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔧 데이터 증강 및 전처리 (적절한 수준으로 완화)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.05,
    height_shift_range=0.05
)

# ✅ 데이터 불러오기
train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ✅ 클래스 이름 저장
class_names = [k.replace('\\', '_') for k in train_data.class_indices.keys()]
with open(os.path.join(SAVE_DIR, "class_names.txt"), "w") as f:
    for name in class_names:
        f.write(name + "\n")
print(f"📝 클래스 목록 저장 완료: {class_names}")

# ✅ 클래스 가중치 계산
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))
print("⚖️ 클래스 가중치:", class_weights)

# ✅ EfficientNetB0 기반 모델 구성
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# 🔧 fine-tuning: 마지막 30개 레이어만 학습
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# ✅ 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ 콜백 설정
checkpoint = ModelCheckpoint(
    os.path.join(SAVE_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# ✅ 학습
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# ✅ 모델 저장
model.save(os.path.join(SAVE_DIR, 'personal_color_model.h5'))
print("✅ 모델 학습 및 저장 완료")
