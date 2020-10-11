import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATASET FROM KERAS AND SPLIT TRAINING AND TEST DATA
data = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = data.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# NORMALIZE THE DATA
X_train = X_train/255.0
X_test = X_test/255.0

# EXAMINING THE TRAINING DATA
'''
# CHECK OUT THE ARRAY FOR THE FIRST TRAINING IMAGE
print(X_train[0])

# CHECK OUT SHAPE OF FIRST TRAINING IMAGE
print(X_train[0].shape)

# PRINT THE FIRST TRAINING IMAGE
plt.imshow(X_train[0], cmap = plt.cm.binary)
plt.show()
'''

# DEFINE AND FIT THE MODEL
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 5)


# EVALUATE THE MODEL
'''
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Tested Acc:', test_acc)
'''

# MAKE A PREDICTION
prediction = model.predict(X_test)

# PRINT PREDICTION FOR FIRST IMAGE
#print(class_names[np.argmax(prediction[0])])

# PREDICT THE FIRST 5 IMAGES

for i in range(5):
    plt.grid(False)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + class_names[y_test[i]])
    plt.title('Prediction: ' +class_names[np.argmax(prediction[i])])
    plt.show()
