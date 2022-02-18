import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Conv1D, Conv2D, Flatten, Dense, Reshape, MaxPooling1D, MaxPooling2D, Dropout, concatenate, Activation, Add
from keras.optimizers import SGD
import tensorflow as tf
import tensorflow_addons as tfa

def build_nn(input_shape1,input_shape2):
	input1 = Input(shape=input_shape1,name='input1')
	A1 = Conv1D(64, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(input1)
	pool11 = MaxPooling1D(pool_size=4)(A1)
	A2 = Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool11)
	pool12 = MaxPooling1D(pool_size=2)(A2)
	A3 = Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool12)
	pool13 = MaxPooling1D(pool_size=2)(A3)
	A4 = Conv1D(16, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool13)
	pool14 = MaxPooling1D(pool_size=2)(A4)

	flattened_wf = Flatten()(pool14)  

	input2 = Input(shape=input_shape2,name='input2')
	B1 = Conv1D(64, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(input2)
	pool21 = MaxPooling1D(pool_size=4)(B1)
	B2 = Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool21)
	pool22 = MaxPooling1D(pool_size=2)(B2)
	B3 = Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool22)
	pool23 = MaxPooling1D(pool_size=2)(B3)
	B4 = Conv1D(16, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool23)
	pool24 = MaxPooling1D(pool_size=2)(B4)

	flattened_fft = Flatten()(pool24) 

	concatted = concatenate([flattened_wf, flattened_fft], axis = 1)

	den1 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(concatted)
	den2 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(den1)
	den3 = Dense(3)(den2)
	pred = Activation('softmax', name='linear')(den3)

	model = Model(inputs=[input1,input2],outputs=[pred])
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy',
		tf.keras.metrics.CategoricalCrossentropy()])
	return model

# plot_model(model,to_file='demo.png',show_shapes=True)

def get_early_stop():
	return EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True)	
