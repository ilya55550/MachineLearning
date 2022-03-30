import re
import os.path
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Input, SimpleRNN
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot as plt


# Формирование выходной прогнозируемой строки
def build_phrase(model_nn, inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)
        pred = model_nn.predict(inp)
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]
        inp_str += d
    return inp_str


with open('train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')
    text = re.sub(r'[^А-я ]', '', text)
num_characters = 34

tokenizer = Tokenizer(num_words=num_characters, char_level=True)
tokenizer.fit_on_texts(text)
print(tokenizer.word_index)

# Преобразование текста в набор векторов
inp_chars = 6
data = tokenizer.texts_to_matrix(text)

# Вычисление размера обучающего множества:
n = data.shape[0] - inp_chars

# Формирование входного тензора и прогнозных значений:
X = np.array([data[i:i + inp_chars, :] for i in range(n)])
Y = data[inp_chars:]

# Создание/загрузка рекуррентной нейронной сети
exists_model = False
if os.path.exists('RNN_model1.h5'):
    exists_model = True
    model1 = load_model('RNN_model1.h5')
else:
    model1 = Sequential()
    model1.add(Input((inp_chars, num_characters)))
    model1.add(SimpleRNN(500, activation='tanh'))
    model1.add(Dense(num_characters, activation='softmax'))
    model1.summary()
    model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history1 = model1.fit(X, Y, batch_size=32, epochs=100)

    # Сохранение обученной модели
    model1.save('RNN_model1.h5')

# Использование
input_symbols = 'Практи'
result_predict = []
result_predict.append(build_phrase(model1, input_symbols))

# Создание/загрузка рекуррентной нейронной сети
if os.path.exists('RNN_model2.h5'):
    model2 = load_model('RNN_model2.h5')
else:
    model2 = Sequential()
    model2.add(Input((inp_chars, num_characters)))
    model2.add(SimpleRNN(500, activation='tanh', return_sequences=True))
    model2.add(SimpleRNN(250, activation='tanh'))
    model2.add(Dense(num_characters, activation='softmax'))
    model2.summary()
    model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history2 = model2.fit(X, Y, batch_size=32, epochs=100)

    # Сохранение обученной модели
    model2.save('RNN_model2.h5')

# Использование
result_predict.append(build_phrase(model2, input_symbols))

# Создание/загрузка рекуррентной нейронной сети
if os.path.exists('RNN_model3.h5'):
    model3 = load_model('RNN_model3.h5')
else:
    model3 = Sequential()
    model3.add(Input((inp_chars, num_characters)))
    model3.add(SimpleRNN(500, activation='tanh', return_sequences=True))
    model3.add(SimpleRNN(250, activation='tanh'))
    model3.add(Dense(num_characters, activation='softmax'))
    model3.summary()
    model3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history3 = model3.fit(X, Y, batch_size=32, epochs=100)

    # Сохранение обученной модели
    model3.save('RNN_model3.h5')

# Использование
result_predict.append(build_phrase(model3, input_symbols))

for i in result_predict:
    print(i)

# сравнение нс по метрикам
if not exists_model:
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 2, 1)
    plt.plot(history1.history['accuracy'])
    plt.title("RNN")
    plt.subplot(2, 2, 2)
    plt.plot(history2.history['accuracy'])
    plt.title("Stacked RNN [500, 200]")
    plt.subplot(2, 2, 3)
    plt.plot(history3.history['accuracy'])
    plt.title("Stacked RNN [1024, 512]")
    plt.show()
