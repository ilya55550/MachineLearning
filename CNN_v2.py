import matplotlib.pyplot as plt
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')
# нормализуем и получаем данные от 0 до 1
x_train /= 255
x_test /= 255
number_of_classes = 10
# унитарная кодировка
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test_cat = np_utils.to_categorical(y_test, number_of_classes)

# отображение датасета
plt.subplot(221)
plt.imshow(x_train[0])
plt.subplot(222)
plt.imshow(x_train[1])
plt.show()

# создание сверточной нейронной сети
if os.path.exists('СNN_model.h5'):
    model = load_model('СNN_model.h5')
else:
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(x_train.shape[1], x_train.shape[2], 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test_cat), epochs=5, batch_size=200)

    plt.plot(history.history['accuracy'])
    plt.show()

    # Сохранение обученной модели
    model.save('СNN_model.h5')

dict_clothes = {
    0: 'Футболка/Клубка',
    1: 'Брюки',
    2: 'Пуловер',
    3: 'Платье',
    4: 'Пальто',
    5: 'Сандал',
    6: 'Рубашка',
    7: 'Кроссовок',
    8: 'Сумка',
    9: 'Ботинок',
}

# Вывод прогноза
for i in range(5):
    x = np.expand_dims(x_test[i], axis=0)
    res = model.predict(x)
    print(res)
    print(f"Значение сети: {dict_clothes[np.argmax(res)]}")

    plt.imshow(x_test[i])
    plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred[:20])
print(y_test[:20])

# Вывод неверных вариантов
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]
expected_value = y_test[~mask]

print("---Вывод неправильно распознанных изображений---")
for i in range(5):
    print("Значение сети: " + dict_clothes[int(p_false[i])])
    print("Правильное значение: " + dict_clothes[int(expected_value[i])])
    plt.imshow(x_false[i])
    plt.show()
