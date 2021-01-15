# importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


# importing datasets
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#training labels set 
Y_train=train_data['label']
# Drop label column
X_train=train_data.drop('label',axis=1)

#normalizing data 
X_train=X_train/255.0
test_data=test_data/255.0

#reshaping data
X_train=X_train.values.reshape(-1,28,28,1)
test_data=test_data.values.reshape(-1,28,28,1)

#spliting train data into training and validation
from sklearn.model_selection import train_test_split

X1_train,Y1_train,X2_train,Y2_train=train_test_split(X_train,Y_train,\
	                                                 test_size=0.1,random_state=2)

#model architecture
model=tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128,activation=tf.nn.relu),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

# compiling the model
model.compile(optimizer='adam',
	          loss='sparse_categorical_crossentropy',
	          metrics=['accuracy'])

#fitting model on training data 
history=model.fit(X1_train,
	              X2_train,
	              batch_size=32,
	              epochs=15,
	              validation_data=(Y1_train,Y2_train))

predicted_classes = model.predict_classes(test_data)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),
                         "Label": predicted_classes})
submissions.to_csv("submission.csv", index=False, header=True)

# Retrieve a list of list results on training and test data
# sets for each training epoch
# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# epochs=range(len(acc)) # Get number of epochs

# #------------------------------------------------
# # Plot training and validation accuracy per epoch
# #------------------------------------------------
# plt.plot(epochs, acc, 'r', "Training Accuracy")
# plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
# plt.title('Training and validation accuracy')
# plt.figure()

# #------------------------------------------------
# # Plot training and validation loss per epoch
# #------------------------------------------------
# plt.plot(epochs, loss, 'r', "Training Loss")
# plt.plot(epochs, val_loss, 'b', "Validation Loss")
# plt.figure()


# # Desired output. Charts with training and validation metrics. No crash :)