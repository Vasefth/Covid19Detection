import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy


# In[ ]:


training_data_generator = ImageDataGenerator(rescale = 1.0/255, 
                                             zoom_range = 0.1, 
                                             rotation_range = 25, 
                                             width_shift_range = 0.05, 
                                             height_shift_range = 0.05)

validation_data_generator = ImageDataGenerator()


# In[ ]:


Directory = 'Covid19-dataset/train'

training_iterator = training_data_generator.flow_from_directory(Directory, class_mode = 'categorical', color_mode = 'grayscale', batch_size = 32)

training_iterator.next()

validation_iterator = validation_data_generator.flow_from_directory(Directory, class_mode = 'categorical', color_mode = 'grayscale', batch_size = 32)


# In[ ]:


def design_model(training_data):
    model = Sequential()
    model.add(tf.keras.Input(shape=(256,256,1)))
    model.add(layers.Conv2D(5,5,strides=3,activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(3,3,strides=1,activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(3,activation='softmax'))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC])
    model.summary()
    return model


# In[ ]:


model = design_model(training_iterator)


# In[ ]:


es = EarlyStopping(monitor='val_auc', mode = 'max', verbose = 1, patience = 20)


# In[ ]:


history = model.fit(training_iterator, steps_per_epoch=training_iterator.samples/32, epochs=5,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples/32,
        callbacks=[es])

