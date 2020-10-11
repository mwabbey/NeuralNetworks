import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#print(tf.__version__)


## MAP THE MNIST DATASET TO TRAINING AND TEST DATA
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


###TRAIN AND FIT THE MODEL THEN SAVE. COMMENTED OUT AFTER MODEL SAVED
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#CHOOSE THE MODEL
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

#FIT THE TRAINING DATA AND DEFINE # OF PASSES
model.fit(x_train, y_train, epochs=3)

#PRINT METRICS FOR TEST DATA
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

#SAVE THE MODEL
model.save('epic_num_reader.model')
'''

## LOAD THE MODEL AND MAKE A PREDICTION
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)
print(np.argmax(predictions[5]))

## PRINT THE TESTED DATA TO CONSOLE
plt.imshow(x_test[5], cmap = plt.cm.binary)
plt.show()

#SHOW THE FIRST NUMBER OF THE TRAINING SET
#plt.imshow(x_train[0], cmap = plt.cm.binary)
#plt.show()
#print(x_train[0])
