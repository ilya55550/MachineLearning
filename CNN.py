import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# отображение датасета
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
plt.show()

X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
# нормализуем и получаем данные от 0 до 1
X_train /= 255
X_test /= 255
number_of_classes = 10
# унитарная кодировка
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

# сравнение сетей с разными параметрами
# # создание сверточной нейронной сети # 1
# model = Sequential()
# model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(number_of_classes, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# history1 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=200)
#
# # создание сверточной нейронной сети # 2
# model = Sequential()
# model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(number_of_classes, activation='softmax'))
#
# model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# print(model.summary())
#
# history2 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=200)
#
# plt.figure(figsize=(10, 20))
# plt.subplot(2, 2, 1)
# plt.plot(history1.history['loss'])
# plt.title("loss nn # 1")
# plt.subplot(2, 2, 2)
# plt.plot(history2.history['loss'])
# plt.title("loss nn # 2")
#
# plt.subplot(2, 2, 3)
# plt.plot(history1.history['accuracy'])
# plt.title("accuracy nn # 1")
# plt.subplot(2, 2, 4)
# plt.plot(history2.history['accuracy'])
# plt.title("accuracy nn # 2")
# plt.show()


# сравнение сетей с разными структурами
# создание сверточной нейронной сети c MaxPooling2D
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history1 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200)

# создание сверточной нейронной сети с AveragePooling2D
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
print(model.summary())

history2 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200)

# создание сверточной нейронной сети с нормализацией
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history3 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200)

# создание сверточной нейронной сети без пулинга

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history4 = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200)

plt.figure(figsize=(10, 20))
plt.subplot(2, 4, 1)
plt.plot(history1.history['loss'])
plt.title("loss nn c MaxPooling2D")
plt.subplot(2, 4, 2)
plt.plot(history2.history['loss'])
plt.title("loss nn с AveragePooling2D")
plt.subplot(2, 4, 3)
plt.plot(history3.history['loss'])
plt.title("loss nn с нормализацией")
plt.subplot(2, 4, 4)
plt.plot(history4.history['loss'])
plt.title("loss nn без пулинга")

plt.subplot(2, 4, 5)
plt.plot(history1.history['accuracy'])
plt.title("accuracy nn c MaxPooling2D")
plt.subplot(2, 4, 6)
plt.plot(history2.history['accuracy'])
plt.title("accuracy nn с AveragePooling2D")
plt.subplot(2, 4, 7)
plt.plot(history3.history['accuracy'])
plt.title("accuracy nn с нормализацией")
plt.subplot(2, 4, 8)
plt.plot(history4.history['accuracy'])
plt.title("accuracy nn без пулинга")
plt.show()
