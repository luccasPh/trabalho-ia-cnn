#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:07:20 2019

@author: lucas
"""

#importa bibliotecas do keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import 7ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

#criando o cnn
def build():
    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=3, input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    
    cnn.add(Flatten())
    
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dense(units=1, activation='sigmoid'))
    
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return cnn

#ler o dataset de imagems
def dataset():
    trainer_data = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
    
    test_data = ImageDataGenerator(rescale=1./255)
    
    trainer_set = trainer_data.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

    test_set = test_data.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
    
    return trainer_set, test_set

#gerar o modelo e carrega a base de treino e teste


print("[INFO] criando o CNN...")
classifier = build()
    
print("[INFO] Carregado o banco de imagems...")
trainer, test = dataset()
    
    #Fitting o CNN com as imagems
print("[INFO] treinado o CNN...")
classifier.fit_generator(trainer,
                             steps_per_epoch=8000,
                             epochs=10,
                             validation_data=test,
                             validation_steps=2000)
    
    #salar a cnn treinada e os pessos
    
model = classifier.to_json()
with open("model.json", "w") as file:
    file.write(model)
    
classifier.save_weights("weights.h5")


classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
test = image.load_img('dataset/cat_or_dog.jpg', target_size = (64,64))
test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)
    
res = model.predict(test)
print("dog") if res[0][0] == 1 else print("cat")


    




