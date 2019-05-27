import numpy as np
from keras.datasets import boston_housing  # данные
from keras.models import Sequential
from keras.layers import Dense

# Задаем seed для повторяемости результатов
np.random.seed(42)

# Загружаем данные
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Стандартизируем данные (необходимо, когда одни данные в еденицах, другие в десятках и т.д.)

# Среднее занчение
mean = x_train.mean(axis=0)
# Стандартное отклонение
std = x_train.std(axis=0)
# Вычитаем среднии значения
x_train = x_train - mean
x_test = x_test - mean
# Делим на стандартое отклонение
x_train = x_train / std
x_test = x_test / std

# Cоздаем нейронную сеть
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))

# (mse - средне квадратичная ошибка, mae - средняя абсолютная ошибка)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Обучаем сеть (batch_size - размер минивыборки(важно для градиента), epochs - сколько эпох для
# одного и того же набора изображений, verbose - печатаем прогресс нейронной сети)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

# оцениваем качество
mse, mae = model.evaluate(x_test, y_test, verbose=0)
print("Средняя абсолютная ошибка", mae)

# Используем сеть
pred = model.predict(x_test)
print("Предсказанная стоимость в тысячах долларов: ", pred[1][0], "\n Настоящая стоимость: ", y_test[1])
