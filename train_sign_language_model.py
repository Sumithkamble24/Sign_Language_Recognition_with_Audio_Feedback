
# import os
# import cv2
# import numpy as np
# import json
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dataset_path = 'dataset1/'  
# img_size = 224

# def preprocess_images(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (img_size, img_size))
#     img = img / 255.0
#     return img

# def load_data(dataset_path):
#     data = []
#     labels = []
#     class_names = sorted(os.listdir(dataset_path))
#     for label, class_name in enumerate(class_names):
#         class_path = os.path.join(dataset_path, class_name)
#         for img_name in os.listdir(class_path):
#             img_path = os.path.join(class_path, img_name)
#             img = preprocess_images(img_path)
#             data.append(img)
#             labels.append(label)
#     return np.array(data), np.array(labels), class_names

# X, y, class_names = load_data(dataset_path)
# y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# datagen.fit(X_train)
# class_indices = {name: idx for idx, name in enumerate(class_names)}
# with open('class_labels.json', 'w') as f:
#     json.dump(class_indices, f)

# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
# base_model.trainable = False

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(256, activation='relu'),
#     Dropout(0.5),   
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(class_names), activation='softmax')
# ])

# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(datagen.flow(X_train, y_train,  batch_size=42), epochs=100,validation_data=(X_test, y_test))
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
# model.save('sign_language_model.h5')


import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

dataset_path = 'dataset'  
img_size = 224

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def load_data(dataset_path):
    data = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = preprocess_image(img_path)
                data.append(img)
                labels.append(label)
            except:
                continue
    return np.array(data), np.array(labels), class_names

X, y, class_names = load_data(dataset_path)
y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

class_indices = {name: idx for idx, name in enumerate(class_names)}
with open('class_labels.json', 'w') as f:
    json.dump(class_indices, f)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save('sign_language_model.h5')
