## Some note on Keras and tensorflow

### Save/load model, plot error figure in keras
See kuzushiji_mnist_cnn.ipynb
### Keras compile setting
model.comile(optimizer='', loss='', metrics='')

#### optimizer
Stochastic gradient descent optimizer ('sgd'), RMSProp optimizer ('rmsprop'), Adam family optimizer('adam', 'adagrad','adadelta', 'adam', 'adamax'), Nesterov Adam optimizer ('nadam')
)
#### loss
" Common choices include mean square error ('mse'), categorical_crossentropy ('categorical_crossentropy' or 'sparse_categorical_crossentropy'), and binary crossentropy ('binary_crossentropy') "

* mean square error : 'mse' or tf.keras.losses.mean_squared_error
* categorical_crossentropy: 'categorical_crossentropy' or tf.keras.losses.categorical_crossentropy
* binary_crossentropy: 'binary_crossentropy' or tf.keras.losses.binary_crossentropy
#### metrics
Mean absolute error (['mae']) or accuracy (['accuracy'])

If labels are encoded as interger, use 'sparse_categorical_crossentropy'

### Convert interger to one-hot coding
y = tensorflow.keras.utils.to_categorical(y, number_class)

### Dense and Conv2D
Conv2D data format :  (samples, channels, new_rows, new_cols)  or  (samples, new_rows, new_cols, channels) 
    
    x = Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))) 
    // input : (n, 28, 28, 1)
    // output: (n, 28, 28 64)

Dense data format : (sample, length)

    x = Dense(64, 'relu', input_shape=(32, )) 
    // input : (n, 32)
    // output: (n, 64)

### Layer setting : Not only 'activation'
    layers.Dense(64, activation='relu', input_shape=(32,))
    layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
    layers.Dense(64, kernel_initializer='orthogonal')
    layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
    layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))