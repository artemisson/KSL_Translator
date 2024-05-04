import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

DATA_PATH = os.path.join(r'KSL 4 signs\KSL_Data') 
actions = np.array(['hello','toothache','fever','headache'])

no_sequences = 30
sequence_length = 30
start_folder = 30

#Data Labelling
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)

#Binary value labelling
y = to_categorical(labels).astype(int)

#Training & Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

log_dir = os.path.join(r'KSL 4 signs\KSL_Logs')
tb_callback = TensorBoard(log_dir=log_dir)
 
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=150, callbacks=[tb_callback])
model.summary()

#Predictions to test Accuracy
res = model.predict(X_test)
#actions[np.argmax(res[4])]
#actions[np.argmax(y_test[4])]

#Save Weights
model.save('KSLmodel.h5')

