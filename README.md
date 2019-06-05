## Jupyter notebook
### Run javascript
See javascript_load_image.ipynb

## Some note on Keras and tensorflow

### Save/load model, plot error figure in keras
See kuzushiji_mnist_cnn.ipynb, mnist_fashion.ipynb
### Keras compile setting
model.compile(optimizer='', loss='', metrics=[''])

#### optimizer
* Stochastic gradient descent optimizer ('sgd')
* RMSProp optimizer ('rmsprop')
* Adam family optimizer('adam', 'adagrad','adadelta', 'adam', 'adamax')
* Nesterov Adam optimizer ('nadam')
)
#### loss
" Common choices include mean square error ('mse'), categorical_crossentropy ('categorical_crossentropy' or 'sparse_categorical_crossentropy'), and binary crossentropy ('binary_crossentropy') "

* mean square error : 'mse' or tf.keras.losses.mean_squared_error
* categorical_crossentropy: 'categorical_crossentropy' or tf.keras.losses.categorical_crossentropy
* binary_crossentropy: 'binary_crossentropy' or tf.keras.losses.binary_crossentropy

If labels are encoded as interger, use 'sparse_categorical_crossentropy'

#### metrics
* Mean absolute error (['mae']) 
* accuracy (['accuracy'])

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
    layers.Dense(64, activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.01))
    layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

### Create Layer and Model manually
See mnist.ipynb

### Use ResNet50V2 or ResNeXt50 on kera 2.2.4

    import keras_applications
    keras_applications.resnext.ResNeXt50(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top = False, weights = 'imagenet', 
        backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils)
        
        
