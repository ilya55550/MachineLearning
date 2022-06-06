### Описание проекта:
В данном проекте предсталено обучение следующих архитектур нейронных сетей:
- НС прямого распространения
- Свёрточные нейронные сети (CNN)
- Рекуррентные нейронные сети (RNN)
- Генеративно-состязательные нейронные сети (GAN)

Проводится сравнение различных параметров НС с учётом различных метрик (mse, accuracy), визуализируется с помощью библиотеки Matplotlib

### Инструменты разработки

**Стек:**
- Python
- Numpy
- Matplotlib
- Sklearn
- Keras
- Tensorflow


## Установка

    Для обучения НС на графическом процессоре потребуется установка дополнительного программного обеспечения (CUDA, cuDNN), ознакомьтесь с документацией: https://www.tensorflow.org/install

    Для обучения на центральном процессоре необходимо прописать - os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## License

[BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)

Copyright (c) 2022-present, ilya55550



