from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, ZeroPadding3D
from keras.layers.merge import concatenate, add
from keras.layers import Dense, Activation, ELU, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from theano import tensor as T
from keras import regularizers
from keras.engine.topology import Layer
import settings
from keras.legacy import interfaces

DROP_RATE=settings.DROP_RATE
EPSILON=settings.EPSILON

class Dropout_uncertain(Layer):
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_uncertain, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, _):
        return self.noise_shape

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=True)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(Dropout_uncertain, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=1.0) 
        return z_avg + K.exp(z_log_var)*eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)

def dice_coef(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = 2. * K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) + 1e-8
    return (intersection / union) 

def recall_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection / (K.sum(y_true_f)+ 1e-8))    

def stable_dice_coef(y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    return dice_coef(y_true, y_pred)

def stable_recall_smooth(y_true, y_pred):
    y_pred = K.sigmoid(y_pred)
    return recall_smooth(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
	"""
	for smoothed dice_coef_loss one can use
	y_true_f = K.clip(K.flatten(y_true), EPSILON, 1-EPSILON)
	"""
	y_pred_f = K.flatten(y_pred)
	y_true_f = K.flatten(y_true) 
	intersection = 2. * K.sum(y_true_f * y_pred_f)
	union = K.sum(y_true_f) + K.sum(y_pred_f) + 1e-8
	return -(intersection / union) 

def stable_nll_loss(y_true, linear_predictor):
	"""
	stable_nll_loss
	"""
	linear_predictor_f = K.flatten(linear_predictor)
	y_true_f = K.flatten(y_true) 

	ll = K.mean(y_true_f * linear_predictor_f - K.log(1.+K.exp(linear_predictor_f)))
	return -ll
	

def _basic_block(n=16, c=1, w=1, h=1):
    def f(input):
        conv_output1 = Conv3D(n, (c, w, h), activation='linear', padding='same', kernel_initializer='he_normal')(input)
        act_output1 = ELU(1.0)(conv_output1)
        conv_output2 = Conv3D(n, (c, w, h), activation='linear', padding='same', kernel_initializer='he_normal')(act_output1)
        return BatchNormalization(axis=1)(conv_output2)
    return f     

def _residual_block(n=16, c=1, w=1, h=1, activation='linear', padding='same'):
    def f(input):
        residual = _basic_block(n, c, w, h)(input)
        return add([input, residual])
    return f

def NConvolution3D(n=16, c=1, w=1, h=1, activation='linear', padding='same'):
    def f(input):
        conv = Conv3D(n, (c, w, h), activation=activation, padding=padding, kernel_initializer='he_normal')(input)
        act_output = ELU(1.0)(conv)
        return BatchNormalization(axis=1)(act_output)
    return f   

class base(object):

	def create_model(self, unused_model_input, **unused_params):
		raise NotImplementedError()


class unet_uncertain_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		up4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_up4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		#Softmax
		conv7 = Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid')(conv6)
		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		"""
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])
		"""
		model.compile(optimizer=Adam(lr=lr), \
			loss='binary_crossentropy', metrics=[dice_coef, 'binary_accuracy', recall_smooth])

		return model


class unet_kendall_2015(base):

	def create_model(self, channel_size, row_size, n_filter, filter_size, lr, TIME_POINT, *unused_params):

		main_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='main_input')
		conv1 = NConvolution3D(n_filter, 3, 3, 3)(main_input)
		conv1 = NConvolution3D(n_filter, 1, 1, 1)(conv1)
		pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
		pool1 = Dropout_uncertain(DROP_RATE)(pool1)

		conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(pool1)
		conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(conv2)
		pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
		pool2 = Dropout_uncertain(DROP_RATE)(pool2)

		conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(pool2)
		conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(conv3)
		conv3 = Dropout_uncertain(DROP_RATE)(conv3)

		up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2], axis=1)
		up4 = NConvolution3D(2*n_filter, 1, 1, 1)(up4)
		conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(up4)
		conv4 = Dropout_uncertain(DROP_RATE)(conv4)

		#Scaled Part
		scaled_input = Input(shape = (TIME_POINT, channel_size, row_size, row_size), dtype='float32', name='aug_input')
		scaled_conv1 = NConvolution3D(n_filter, 3, 3, 3)(scaled_input)
		scaled_conv1 = NConvolution3D(n_filter, 1, 1, 1)(scaled_conv1)
		scaled_pool1 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv1)
		scaled_pool1 = Dropout_uncertain(DROP_RATE)(scaled_pool1)

		scaled_conv2 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_pool1)
		scaled_conv2 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_conv2)
		scaled_pool2 = MaxPooling3D(pool_size=(2, 2, 2))(scaled_conv2)
		scaled_pool2 = Dropout_uncertain(DROP_RATE)(scaled_pool2)

		scaled_conv3 = NConvolution3D(3*n_filter, 3, 3, 3)(scaled_pool2)
		scaled_conv3 = NConvolution3D(3*n_filter, 1, 1, 1)(scaled_conv3)
		scaled_conv3 = Dropout_uncertain(DROP_RATE)(scaled_conv3)        

		scaled_up4 = concatenate([UpSampling3D(size=(2, 2, 2))(scaled_conv3), scaled_conv2], axis=1)
		scaled_up4 = NConvolution3D(2*n_filter, 1, 1, 1)(scaled_up4)
		scaled_conv4 = NConvolution3D(2*n_filter, 3, 3, 3)(scaled_up4)
		scaled_conv4 = Dropout_uncertain(DROP_RATE)(scaled_conv4)

		#Merge two parts
		up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv1, \
		                UpSampling3D(size=(2, 2, 2))(scaled_conv4), scaled_conv1], axis=1)
		conv5 = NConvolution3D(3*n_filter, 1, 1, 1)(up5)
		conv5 = NConvolution3D(3*n_filter, 3, 3, 3)(conv5) 
		conv5 = Dropout_uncertain(DROP_RATE)(conv5)

		conv6 = NConvolution3D(2*n_filter, 3, 3, 3)(conv5)
		conv6 = Dropout_uncertain(DROP_RATE)(conv6)

		# sampling normal
		z_avg=Conv3D(1, (1, 3, 3), padding='same', activation='linear')(conv6)
		z_log_var=Conv3D(1, (1, 3, 3), padding='same', activation='linear')(conv6)

		z=SampleNormal()([z_avg, z_log_var])
		# conv7=Activation('sigmoid')(z)
		conv7 = z

		model = Model(inputs=[main_input, scaled_input], outputs=conv7)
		"""
		model.compile(optimizer=Adam(lr=lr), \
			loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy', recall_smooth])
		"""
		model.compile(optimizer=Adam(lr=lr), \
			loss=stable_nll_loss, metrics=[stable_dice_coef, stable_recall_smooth])

		return model

