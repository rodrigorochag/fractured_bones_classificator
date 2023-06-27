import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Definir a arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Carregar os dados
X_train = np.load('/home/netune/Documents/Bones/dataset/train/fractured')
y_train = np.load('/home/netune/Documents/Bones/dataset/train/"not fractured"')

# Reshape dos dados para ajustar a escala de cinza
X_train = np.reshape(X_train, (X_train.shape[0], 224, 224, 1))

# Treinar a CNN
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Avaliar o modelo em dados de teste
X_test = np.load('/home/netune/Documents/Bones/dataset/val/fractured')
y_test = np.load('/home/netune/Documents/Bones/dataset/val/"not fractured"')

# Reshape dos dados de teste
X_test = np.reshape(X_test, (X_test.shape[0], 224, 224, 1))

loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)
