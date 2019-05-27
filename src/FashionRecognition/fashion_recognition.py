from tensorflow.python.keras.datasets import fashion_mnist # модель для работы с надором данных
from tensorflow.python.keras.models import Sequential # модель для представления нейронной сети, в которой нейроны
# идут последовательно
from tensorflow.python.keras.layers import Dense # тип слоя полносвязной нейронной сети
from tensorflow.python.keras import utils # для приведения данных в необходимый формат в для keras
import numpy as np

# Загружаем данные
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений (полносвязная нейронная сеть не может работать с 2D данными (28х28),
# поэтому преобразовываем в одномерный массив (60000 изображений, 784=28*28 пикселей в каждом))
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Нормализация данных
x_train = x_train / 255
x_test = x_test / 255


# Преобразуем метки в категории
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Названия классов (чтобы выводить названия)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']


# Создаем нейронную сеть

# Создаем последовательную модель
model = Sequential()

# Добавляем слои
# Входной слой (800 - кол-во нейронов, input_dim - кол-во входов в каждый нейрон)
model.add(Dense(800, input_dim=784, activation="relu"))
# Выходной слой
model.add(Dense(10, activation="softmax"))

# Компилируем модель (loss - ф-ия ошибки, SGD - стахост. градиентный спуск, metrics - метрика качества)
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# print(model.summary()) # печатаем параметры сети


# Обучаем сеть (batch_size - размер минивыборки(важно для градиента), epochs - сколько эпох для
# одного и того же набора изображений, verbose - печатаем прогресс нейронной сети)
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

# Оцениваем качество сети на тестовых данных
score = model.evaluate(x_test, y_test, verbose=1)

# Доля правильных ответов
print("Доля прав ответов: ", round(score[1]*100, 4))


# Распознаем предметы данных
predictions = model.predict(x_test)

# Выводим номер класса, предсказанный нейронной сетья
print(np.argmax(predictions[0]))

# Выводим правильный номер класса
print(np.argmax(y_test[0]))
