import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten


def param_neural_network(count_dense, count_neuron_in_layer):
    """count_neuron_in_layer список в соотвествии со слоями"""

    if type(count_neuron_in_layer[0]) == type(list()):
        count_neuron_in_layer = count_neuron_in_layer[0]
    temp = [Dense(count_neuron_in_layer[_], activation='relu') for _ in range(count_dense)]
    model = keras.Sequential([Flatten(input_shape=(28, 28, 1))] + temp + [Dense(10, activation='softmax')])
    return model


def param_fit(model, x_train, y_train_cat, count_epochs):
    history = model.fit(x_train, y_train_cat, batch_size=32, epochs=count_epochs, validation_split=0.2)
    return model, history


def check_type(object_nn):
    return len(object_nn) if type(object_nn) == type(list()) else 1


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)

plt.show()

# list_count_neuron = [64, 256, 512, 1024]
# list_count_neuron = [512, [512, 256], [512, 256, 256]]
list_count_neuron = []
for i in range(1, 10):
    if i == 1:
        list_count_neuron.append(64)
    else:
        list_count_neuron.append([64] * i)

count_epochs = 5
res_model = []

# основной цикл обучения
for i in range(len(list_count_neuron)):
    print(list_count_neuron[i])
    print(check_type(list_count_neuron[i]))
    model = param_neural_network(check_type(list_count_neuron[i]), [list_count_neuron[i]])

    # print(model.summary())  # вывод структуры НС в консоль

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['mse', 'accuracy'])

    model, history = param_fit(model, x_train, y_train_cat, count_epochs)
    res_model.append(history)

# print(f"evaluate: {model.evaluate(x_test, y_test_cat)[0]}")

# Сравнение метрик
plt.figure(figsize=(10, 20))
for i in range(len(list_count_neuron)):
    plt.subplot(2, math.ceil(len(list_count_neuron) / 2), i + 1)
    plt.plot(res_model[i].history['mse'])
    plt.title(f"Кол-во нейронов/слоёв: {list_count_neuron[i]}/{check_type(list_count_neuron[i])}")
plt.show()

plt.figure(figsize=(10, 20))
for i in range(len(list_count_neuron)):
    plt.subplot(2, math.ceil(len(list_count_neuron) / 2), i + 1)
    plt.plot(res_model[i].history['accuracy'])
    plt.title(f"Кол-во нейронов/слоёв: {list_count_neuron[i]}/{check_type(list_count_neuron[i])}")
plt.show()

# Сохранение обученной модели
model.save('model.h5')

# Вывод верных вариантов
for i in range(2):
    x = np.expand_dims(x_test[i], axis=0)
    res = model.predict(x)
    print(res)
    print(f"Значение сети: {np.argmax(res)}")

    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# Вывод неверных вариантов
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
p_false = pred[~mask]

print(x_false.shape)

print("---Вывод неправильно распознанных изображений---")
for i in range(2):
    print("Значение сети: " + str(p_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
