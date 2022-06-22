import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
# Đọc dữ liệu
bodyswing_df = pd.read_csv("SWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")
X = []
y = []
no_of_timesteps = 10
dataset = bodyswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)
dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
X=X.reshape(1182, 1320)
X = X.astype('float32')
y=to_categorical(y, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = Sequential()
model.add(Dense(512, kernel_initializer='normal', activation='relu', input_shape=(1320,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid')) 
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test)) 
model.save("model.h5")
y_pred = model.predict(X_test)
print(y_pred)
score = model.evaluate(X_test, y_pred, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'test'], loc='upper left')
plt.show()
