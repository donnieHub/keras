import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.datasets import mnist

from tensorflow import keras

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


class Mnist:

    def test_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        assert x_train.shape == (60000, 28, 28)
        assert x_test.shape == (10000, 28, 28)
        assert y_train.shape == (60000,)
        assert y_test.shape == (10000,)
        print("x_train.shape", x_train.shape)
        print("x_test.shape ", x_test.shape)
        print("y_train.shape ", y_train.shape)
        print("y_test.shape ", y_test.shape)

        # Выводим картинки на экран
        # plt.figure(figsize=(10,5))
        # for i in range(25, 50):
        #     plt.subplot(5, 5, i+1-25)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(x_train[i], cmap=plt.cm.binary)
        #
        # plt.show()

        # нормализуем (от 0 до 1)
        x_train = x_train / 255
        x_test = x_test / 255
        # делем на вектора по 10 длинной
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_test_cat = keras.utils.to_categorical(y_test, 10)

        # Создаем структуру нейросети
        model = keras.Sequential([Flatten(input_shape=(28, 28, 1)),

                                  Dense(128, activation='relu6'),
                                  Dense(10, activation='softmax')])

        print(model.summary())

        # Конфигурируем нейросеть для тренировки
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Обучаем нейросеть
        his = model.fit(x_train, y_train_cat, batch_size=30, epochs=5, validation_split=0.0001)

        print("-------------------------------------------------------------------------------")
        # Проверяем что сеть обучилась на тестовой выборке
        model.evaluate(x_test, y_test_cat)

        # Тестируем одну картинку
        # pic_num = 10
        # pic = np.expand_dims(x_test[pic_num], axis=0)
        # result = model.predict(pic)
        #
        # print(result)
        # print(f"Распознаная цифра: {np.argmax(result)}")
        #
        # plt.imshow(x_test[pic_num], cmap=plt.cm.binary)
        # plt.show()

        # Расспознаем всю тестовую выборку
        predict = model.predict(x_test)
        predict_max = np.argmax(predict, axis=1)

        print(predict_max.shape)
        print(predict_max[:20])
        print(y_test[:20])

        # Выделение неверных вариантов
        right_result = predict_max == y_test

        x_wrong_result = x_test[~right_result]
        pred_wrong_result = predict_max[~right_result]

        print(x_wrong_result.shape)

        plt.plot(his.history['loss'])
        plt.plot(his.history['val_loss'])
        plt.show()

        # # Вывод первых 5 неверных результатов
        # plt.figure(figsize=(10, 5))
        # for i in range(50, 75):
        #     print("Неверно распознаные результаты: " + str(pred_wrong_result[i]))
        #     plt.subplot(5, 5, i + 1 - 50)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(x_wrong_result[i], cmap=plt.cm.binary)
        #     plt.title(f"Image {pred_wrong_result[i]}")  # add a title to the subplot
        # plt.show()