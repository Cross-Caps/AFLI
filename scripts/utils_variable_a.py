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


## Note: Implementation of custom layer with scaled-htanh activation function. Replace scaled-htanh with function of your choice
#Keras callbacks are used to change the value of parameter 'a'.



# Training routine for network with Gaussian weights
def train(X_train, X_test, y_train, y_test, dim, width, depth, act, var_w2, var_b2, q, opt, nb_epochs, batch_size, trial):
    
	# Weight initialization
    def input_init_weights(shape, dtype=None):
        return K.variable(np.sqrt(var_w2/dim)*np.random.randn(shape[0], shape[1]))

    def init_weights(shape, dtype=None):
        return K.variable(np.sqrt(var_w2/width)*np.random.randn(shape[0], shape[1]))
      
    def init_bias(shape, dtype=None):
        return K.variable(np.sqrt(var_b2)*np.random.randn(shape[0]))
    
    
    # Scaled-htanh activation   
    def S_htanh(x,a=1):
      return tf.where(x>a,a, tf.where(x>-a,x,-a))
      
    # Custom activation layer  
    class custom_act(Layer):

      def __init__(self, beta=1.0, trainable=False, **kwargs):
          super(custom_act, self).__init__(**kwargs)
          self.supports_masking = True
          self.beta = beta
          self.trainable = trainable

      def build(self, input_shape):
          self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
          if self.trainable:
              self._trainable_weights.append(self.beta_factor)

          super(custom_act, self).build(input_shape)

      def call(self, inputs, mask=None):
          return S_htanh(inputs, self.beta_factor)   # call to scale-htanh activation function

      def get_config(self):
          config = {'beta': self.get_weights()[0] if self.trainable else self.beta,
                  'trainable': self.trainable}
          base_config = super(custom_act, self).get_config()
          return dict(list(base_config.items()) + list(config.items()))

      def compute_output_shape(self, input_shape):
          return input_shape
    
    
    # Custom callback to update parameter 'a'   
    class CustomCallback(keras.callbacks.Callback):
      def __init__(self, a_val):
        self.a_val = a_val
      def on_epoch_begin(self, epoch, logs=None):
        if epoch <80:
          for layer in model.layers:
            if layer.__class__.__name__ == 'HtanhA':
              w=layer.get_weights()
              w=[(x-x+self.a_val[epoch]) for x in w]   # (x-x) is done due numerical precision issues
              layer.set_weights(w)
    
    x_train = np.sqrt(q) * X_train
    x_test = np.sqrt(q) * X_test
    
    num_classes = 10
    
	# Model definition
    def get_model(a=1):
        model = Sequential()
        model.add(Dense(width, input_shape=(dim,), kernel_initializer=input_init_weights, bias_initializer=init_bias))
        model.add(custom_act(beta=a, trainable=False))
        for i in range(depth-1):
            model.add(Dense(width, kernel_initializer=init_weights, bias_initializer=init_bias))
            model.add(custom_act(beta=a, trainable=False))
        model.add(Dense(num_classes, kernel_initializer=init_weights, bias_initializer=init_bias, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model  

       
    
    # Build the model
    model = get_model(a=5)   # Start with a higher value of parameter 'a' i.e., a more linear network
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Variable 'a' values
    a_val=np.flip(np.linspace(2,10,80,endpoint=True))   
    
    # Fit the model
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, batch_size=batch_size, verbose=0,callbacks=[CustomCallback1()])
    # Save History
    file_name=act+'_b='+str(var_b2)+'_depth='+str(depth)+'_trial'+str(trial)
    np.save('actV1/'+file_name,history.history)










   
    
    
    
    
