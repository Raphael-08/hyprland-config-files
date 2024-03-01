import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

new_model = tf.keras.models.load_model("/home/mrv/coding/ml/epic_num_reader.h5")


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def draw(n):
    plt.imshow(n, cmap=plt.cm.binary)
    plt.show()


os.system("clear")

predictions = new_model.predict([x_test])

# while True:
for i in range(1000):
    # i = int(input("enter a number: "))

	if(y_test[i]!=np.argmax(predictions[i])):
		print(i)
		print("label -> ", y_test[i])
		print("prediction -> ", np.argmax(predictions[i]))

    # draw(x_test[i])
