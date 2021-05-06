# -*- coding: utf-8 -*-
import pickle
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dense, Lambda, Flatten, Activation


# Training routine for network with Gaussian weights
def train(X_train, X_test, y_train, y_test, dim, width, depth, act, var_w2, var_b2, q, opt, nb_epochs, batch_size, trial):
    
	# Weight initialization
    def input_init_weights(shape, dtype=None):
        return K.variable(np.sqrt(var_w2/dim)*np.random.randn(shape[0], shape[1]))

    def init_weights(shape, dtype=None):
        return K.variable(np.sqrt(var_w2/width)*np.random.randn(shape[0], shape[1]))
      
    def init_bias(shape, dtype=None):
        return K.variable(np.sqrt(var_b2)*np.random.randn(shape[0]))
    
    
    # Custom scaled-activation functions with fixed hardcoded value of parameter 'a'  
    def S_htanh(x):
      a=1.0
      return tf.where(x>a,a, tf.where(x>-a,x,-a))
      
    def S_sin(x):
      a=1.0
      return tf.where(x>a,(a-1)+K.sin(tf.constant(np.pi)/2 - a +x), tf.where(x>-a,x,-(a-1)-K.sin(tf.constant(np.pi)/2 - a - x)))
      
    def S_exp(x):
      a=1.0
      return tf.where(x>a,(a-1)+K.exp(-x + a), tf.where(x>-a,x,-(a-1)-K.exp(x + a) ))

    def S_saw(x):
      a=1.0
      return tf.where(x<(-2.0*a),0.0, tf.where(x>(2.0*a), 0.0, tf.where(x<-a, -x-(2.0*a) , tf.where( x>a,-x+(2.0*a),x )   )  ) )    
    
    
    get_custom_objects().update({'S_htanh': Activation(S_htanh)})
    get_custom_objects().update({'S_sin': Activation(S_sin)})
    get_custom_objects().update({'S_exp': Activation(S_exp)})
    get_custom_objects().update({'S_saw': Activation(S_saw)})
    
    x_train = np.sqrt(q) * X_train
    x_test = np.sqrt(q) * X_test
    
    num_classes = 10
    
	# Model definition
    def get_model():
        model = Sequential()
        model.add(Dense(width, input_shape=(dim,), kernel_initializer=input_init_weights, bias_initializer=init_bias,activation=act))
        for i in range(depth-1):
            model.add(Dense(width, kernel_initializer=init_weights, bias_initializer=init_bias,activation=act))
        model.add(Dense(num_classes, kernel_initializer=init_weights, bias_initializer=init_bias, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model  
    
    
    # Build the model
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    # Fit the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size, verbose=0)
    # Save History
    file_name=act+'_b='+str(var_b2)+'_depth='+str(depth)+'_trial'+str(trial)
    np.save('exp/'+file_name,history.history)





# Training routine for network with orthogonal weights    
def trainO(X_train, X_test, y_train, y_test, dim, width, depth, act, var_w2, var_b2, q, opt, nb_epochs, batch_size, trial):

        
    def init_bias(shape, dtype=None):
        return K.variable(np.sqrt(var_b2)*np.random.randn(shape[0]))
    

    
    # Custom scaled-activation functions with fixed hardcoded value of parameter 'a'  
    def S_htanh(x):
      a=1.0
      return tf.where(x>a,a, tf.where(x>-a,x,-a))
      
    def S_sin(x):
      a=1.0
      return tf.where(x>a,(a-1)+K.sin(tf.constant(np.pi)/2 - a +x), tf.where(x>-a,x,-(a-1)-K.sin(tf.constant(np.pi)/2 - a - x)))
      
    def S_exp(x):
      a=1.0
      return tf.where(x>a,(a-1)+K.exp(-x + a), tf.where(x>-a,x,-(a-1)-K.exp(x + a) ))

    def S_saw(x):
      a=1.0
      return tf.where(x<(-2.0*a),0.0, tf.where(x>(2.0*a), 0.0, tf.where(x<-a, -x-(2.0*a) , tf.where( x>a,-x+(2.0*a),x )   )  ) )
    
	
    x_train = np.sqrt(q) * X_train
    x_test = np.sqrt(q) * X_test
    
    num_classes = 10

	# Model definition
    def get_model():
        model = Sequential()
        model.add(Dense(width, input_shape=(dim,), kernel_initializer=initializers.Orthogonal(gain=np.sqrt(var_w2), seed=None), bias_initializer=init_bias,activation=act))
        for i in range(depth-1):
            model.add(Dense(width, kernel_initializer=initializers.Orthogonal(gain=np.sqrt(var_w2), seed=None), bias_initializer=init_bias,activation=act))
        model.add(Dense(num_classes, kernel_initializer=initializers.Orthogonal(gain=np.sqrt(var_w2), seed=None), bias_initializer=init_bias, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
       
    
    # Build the model
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
    # Fit the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size, verbose=0)
    # Save History
    file_name=act+'_b='+str(var_b2)+'_depth='+str(depth)+'_trial'+str(trial)
    np.save('expO/'+file_name,history.history)
    
    
    
   
    
    
    
    
