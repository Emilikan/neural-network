from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import utils

from tensorflow.python.keras.models import load_model  # для загрузки модели из файла
from tensorflow.python.keras.models import model_from_json  # для загрузки архитектуры сети из json

# подготавливаем данные и модель

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train / 255
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)


# Сохраняем модель
model.save('fashion_mnist_dense.h5')

# Загружаем модель из файла
newModel = load_model('fashion_mnist_dense.h5')

# Используем сеть
prediction = newModel.predict(x_train)


# Сохранение только весов модели
model.sample_weights('fashion_mnist_weights.h5')

# Загрузка весов в модель (при этом надо будет создать архитектуру нс)
newModelWeights = Sequential()
newModelWeights.load_weights('fashion_mnist_weights.h5')


# Сохранение только архитектуры модели в формате json (затем json можно сохранить локально)
json_string = model.to_json()

# Загружаем архитектуру сети из json (нейронная сеть будет не обучена)
newModelFromJson = model_from_json(json_string)
