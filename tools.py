import random
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas as pd
from sklearn import cluster, datasets
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier


def index_elem(array):
    """Найти индексы ненулевых элементов в заданном массиве"""
    return np.nonzero([1, 2, 0, 0, 4, 0])


def create_one_matrix(n):
    """Создать nxn единичную матрицу"""
    return np.ones((n, n))


def create_random_matrix(n):
    """Создать массив nxnxn со случайными значениями"""
    return np.random.randint(0, 3, (n, n, n))


def create_vector_ones(n):
    """Создать вектор (одномерный массив) размера n, заполненный нулями"""
    return np.ones(n)


def create_vector_zeros(n):
    """Создать вектор размера 1n, заполненный нулями, но пятый элемент равен 1"""
    matrix = np.zeros(n)
    try:
        matrix[5] = 1
    except IndexError:
        print("Вектор состоит из менее 6 элементов, поэтому невозможно изменить 5 элемент")
    return matrix


def create_vector_randint(n, m, ln):
    """Создать вектор со значениями от n до m"""
    return np.random.randint(n, m, ln)


def reverse_array(array):
    """Развернуть одномерный массив (первый становится последним)"""
    return array[::-1]


def create_matrix(n):
    """Создать матрицу с 0 внутри, и 1 на границах"""
    array = np.zeros((n, n))
    array[0::n - 1, :] = 1
    array[:, 0::n - 1] = 1
    return array


def create_chess_matrix(n):
    """Создать nxn матрицу и заполнить её в шахматном порядке"""
    array = np.zeros((n, n))
    array[::2, ::2] = 1
    array[1::2, 1::2] = 1
    return array


def change_max_value(array):
    """Заменить максимальный элемент на ноль"""
    print(f"До изменения:\n {array}")
    sum_index = np.argmax(array)
    print(f"\nmax elem: {array.max()}")
    array[sum_index // len(array)][sum_index % len(array)] = 0
    print(f"\nПосле изменения:\n {array}")
    return array


def search_nearest_value(array, value):
    """Найти ближайшее к заданному значению число в заданном массиве"""
    index = (np.abs(array - value)).argmin()
    print(f'{array}\nБлижайшее значение к {value}: {array[index]}')
    return array[index]


def calculate_sum_axis(array):
    """Дан трехмерный массив, посчитать сумму по последней оси"""
    print(f'{array}\n------------------')
    return array.sum(axis=len(np.shape(array)) - 1)


def subtract_the_average(array):
    """Отнять среднее из каждой строки в матрице"""
    print(f'{array}\n----')
    return array - array.mean(axis=1, keepdims=True)


array = [1, 3, 0, 5, 0, 1, 10, 0]  # 1
print(index_elem(array))

print(create_one_matrix(3))  # 2

array = create_random_matrix(3)  # 3
print(array)
print(f"\nshape: {np.shape(array)}")

array = create_vector_ones(4)  # 4
print(array)
print(f"\nshape: {np.shape(array)}")

print(create_vector_zeros(6))  # 5

print(create_vector_randint(0, 6, 10))  # 6

array = np.array([1, 2, 3, 4, 5])  # 7
print(reverse_array(array))

print(create_matrix(6))  # 8

print(create_chess_matrix(6))  # 9

array = np.random.randint(0, 100, (4, 4))  # 10
change_max_value(array)

array = np.random.randint(0, 100, 10)  # 11
search_nearest_value(array, 20)

array = np.random.randint(0, 100, (4, 4, 4))  # 12
print(calculate_sum_axis(array))

array = np.array([[0, 100, 3], [0, 5, 6], [0, 7, 8]])  # 13
"""Определить, есть ли в двумерном массиве нулевые столбцы"""
define_empty_columns = lambda array: True if 0 in array.max(axis=0) else False
print(define_empty_columns(array))

array = np.random.randint(0, 100, (3, 3))  # 14
print(subtract_the_average(array))


# Matplotlib_№1

def sin_func(x):
    return np.sin(3 * x) / x if x != 0 else np.sin(3 * x)


x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(10, 5))
plt.plot(x, [sin_func(i) for i in x], label=r'$y=\sin(3*x)/x$')
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig('figure.png')
plt.show()


# Matplotlib_№2

def sin_func3(x):
    return np.sin(3 * x) / x if x != 0 else 3


def sin_func2(x):
    return np.sin(2 * x) / x if x != 0 else 2


def sin_func(x):
    return np.sin(x) / x if x != 0 else 1


x = np.arange(-360, 361)
ax = plt.gca()
ax.plot(x, [sin_func3(np.deg2rad(i)) for i in x], 'b', label=r'$y=\sin(3*x)/x$')
ax.plot(x, [sin_func2(np.deg2rad(i)) for i in x], 'r', label=r'$y=\sin(2*x)/x$')
ax.plot(x, [sin_func(np.deg2rad(i)) for i in x], 'g', label=r'$y=\sin(x)/x$')

plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)

x_points = np.array([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
x_labels = ['-$2\u03C0$', '-$\u03C0$', '$0$', '$\u03C0$', '$2\u03C0$']
plt.xticks(np.rad2deg(x_points), x_labels)

y_points = np.array(list(range(-1, 4)))
plt.yticks(y_points)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# ax.grid(True)
plt.show()

# Matplotlib_№3

data = sorted([random.randint(0, 1000) for i in range(10)], reverse=True)

pylab.figure(figsize=(10, 20))
pylab.subplot(2, 2, 1)

pylab.pie(data, autopct='%.1f', radius=1.1, explode=[0.15] + [0 for _ in range(len(data) - 1)])
pylab.legend(bbox_to_anchor=(-0.5, 0.1, 0.25, 0.25), loc='lower left', labels=data)
pylab.title("Круговая диаграмма")

pylab.subplot(2, 2, 2)
pylab.bar([x + 0.05 for x in range(len(data))], [d * 0.9 for d in data], width=0.2, color='red', alpha=0.7, zorder=2)
pylab.bar([x + 0.3 for x in range(len(data))], data, width=0.2, color='blue', alpha=0.7, zorder=2)
pylab.title("Стобчатая диаграмма")

pylab.subplot(2, 2, 3)
pylab.hist(data, edgecolor='black', alpha=0.8)
pylab.title("Гистограмма значений")

pylab.show()

# Sklearn
# 4.1 Линейная регрессия

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)

# 4.2 k-Классификатор ближайшего соседа

# загрузка датасета
iris = datasets.load_iris()
# Создание и обучение классификатора ближайшего соседа
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
# прогнозирование для входных данных
result = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(iris.data)
print(result)

# 4.3  К-средство кластеризации

# load data
iris = datasets.load_iris()
# create clusters for k=3
k = 3
k_means = cluster.KMeans(k)
# fit data
k_means.fit(iris.data)
print(k_means.labels_[::10])
print(iris.target[::10])

# 4.4 Обучение на основе собственного датасета

df = pd.read_excel('data.xlsx', index_col=0)
df.head()

X = df.drop('target', axis=1)
y = df['target']

model = RandomForestClassifier()

model.fit(X, y)

example = {'age': [31],
           'city_Екатеринбург': [0],
           'city_Киев': [0],
           'city_Краснодар': [1],
           'city_Минск': [0],
           'city_Москва': [0],
           'city_Новосибирск': [0],
           'city_Омск': [0],
           'city_Петербург': [0],
           'city_Томск': [0],
           'city_Хабаровск': [0],
           'city_Ярославль': [0],
           'family_members': [0],
           'salary': [130000],
           'transport_preference_Автомобиль': [1],
           'transport_preference_Космический корабль': [0],
           'transport_preference_Морской транспорт': [0],
           'transport_preference_Поезд': [0],
           'transport_preference_Самолет': [0],
           'vacation_preference_Архитектура': [0],
           'vacation_preference_Ночные клубы': [0],
           'vacation_preference_Пляжный отдых': [0],
           'vacation_preference_Шопинг': [1]}

example_df = pd.DataFrame(example)
model.predict(example_df)
