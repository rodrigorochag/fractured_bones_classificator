import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Definir a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Carregar os dados da primeira classe
class1_folder_path = '/home/netune/Documents/Bones/fractured_bones_classificator/dataset/train/fractured'
class1_image_paths = glob.glob(class1_folder_path + '/*.jpg')

X_train_class1 = []
y_train_class1 = []

for path in class1_image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    X_train_class1.append(img)
    y_train_class1.append(0)  # Supondo que a primeira classe seja rotulada como 0

# Carregar os dados da segunda classe
class2_folder_path = '/home/netune/Documents/Bones/fractured_bones_classificator/dataset/train/not_fractured'
class2_image_paths = glob.glob(class2_folder_path + '/*.jpg')

X_train_class2 = []
y_train_class2 = []

for path in class2_image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    X_train_class2.append(img)
    y_train_class2.append(1)  # Supondo que a segunda classe seja rotulada como 1

# Combinar os dados de ambas as classes
X_train = np.array(X_train_class1 + X_train_class2)
y_train = np.array(y_train_class1 + y_train_class2)

# Normalizar os valores dos pixels entre 0 e 1
X_train = X_train.astype('float32') / 255

# Adicionar uma dimensão extra para os canais das imagens (no caso de imagens em escala de cinza)
X_train = np.expand_dims(X_train, axis=-1)

# Treinar a CNN
model.fit(X_train, y_train, epochs=2, batch_size=32)
model.save("imagens.h5")
# Continuação do código para avaliar o modelo em dados de teste...
