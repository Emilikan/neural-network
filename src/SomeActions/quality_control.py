from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import utils

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


# Оценка качества

# Подготавливаем тестовую выборку
x_test = x_test.reshape(10000, 784)
x_test = x_test / 255

# Оцениваем качество сети на тестовых данных
score = model.evaluate(x_test, y_test, verbose=1)

# Доля правильных ответов
print("Доля правильных ответов:", round(score[1]*100, 4))
