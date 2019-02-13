from keras import backend as K

def dice_coefficient(y_true, y_pred):
    """
    A statistic used for comparing the similarity of two samples. Here binary segmentations.

    Args:
        y_true (numpy.array): the true segmentation
        y_pred (numpy.array): the predicted segmentation

    Returns:
        (float) returns a number from 0. to 1. measuring the similarity y_true and y_pred
    """
    y_true_f=K.flatten(y_true)
    mu     = y_pred[:,:,:,0]
    y_pred_f=K.flatten(mu)
    intersection=K.sum(y_true_f*y_pred_f)
    smooth=1e-5
    return (2*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)

def recall_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection / (K.sum(y_true_f) + K.epsilon()))  

def precision_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection / (K.sum(y_pred_f) + K.epsilon()))  

def UNet(N_filters=32):
    '''
    U net architecture (down/up sampling with skip architecture)
    '''
    from keras.layers import Input, Dropout, Activation, Add, BatchNormalization, Conv2D, Concatenate, UpSampling2D
    def Conv2DReluBatchNorm(n_filters, kernel_size, strides, inputs):
        x = Conv2D(n_filters, (1,1), strides=1, padding='same', 
                   kernel_initializer='he_normal',
                   activation='elu')(inputs)
        x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', 
                   kernel_initializer='he_normal',
                   activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.25)(x, training=True) # training + test-time dropout!
        return x

    inputs = Input((None, None, 3))
    
    layer1 = Conv2DReluBatchNorm(N_filters,  (5, 5), (1,1), inputs)
    layer2 = Conv2DReluBatchNorm(2*N_filters,  (5, 5), (2,2), layer1)
    layer3 = Conv2DReluBatchNorm(4*N_filters, (3, 3), (2,2), layer2)
    layer4 = Conv2DReluBatchNorm(8*N_filters, (3, 3), (2,2), layer3)
    layer5 = Conv2DReluBatchNorm(16*N_filters, (3, 3), (2,2), layer4)

    merge6 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer5), layer4])
    layer6 = Conv2DReluBatchNorm(8*N_filters, (3, 3), (1,1), merge6)
    
    merge7 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer6), layer3])
    layer7 = Conv2DReluBatchNorm(4*N_filters, (3, 3), (1,1), merge7)

    merge8 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer7), layer2])
    layer8 = Conv2DReluBatchNorm(2*N_filters, (3, 3), (1,1), merge8)

    merge9 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer8), layer1])
    layer9 = Conv2DReluBatchNorm(N_filters, (3, 3), (1,1), merge9)

    output = Conv2D(1, (1, 1), strides=(1,1), activation='sigmoid', name='output')(layer9)    

    from keras.models import Model
    return Model(inputs=inputs, outputs=output)



def Small_UNet(N_filters=32):
    '''
    U net architecture (down/up sampling with skip architecture)
    '''
    from keras.layers import Input, Dropout, Activation, Add, BatchNormalization, Conv2D, Concatenate, UpSampling2D
    def Conv2DReluBatchNorm(n_filters, kernel_size, strides, inputs):
        x = Conv2D(n_filters, (1,1), strides=1, padding='same', 
                   kernel_initializer='he_normal',
                   activation='elu')(inputs)
        x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', 
                   kernel_initializer='he_normal',
                   activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.25)(x, training=True) # training + test-time dropout!
        return x

    inputs = Input((None, None, 3))
    
    layer1 = Conv2DReluBatchNorm(N_filters,  (5, 5), (1,1), inputs)
    layer2 = Conv2DReluBatchNorm(2*N_filters,  (5, 5), (2,2), layer1)
    layer3 = Conv2DReluBatchNorm(4*N_filters, (3, 3), (2,2), layer2)
    layer4 = Conv2DReluBatchNorm(8*N_filters, (3, 3), (2,2), layer3)
    
    merge5 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer4), layer3])
    layer5 = Conv2DReluBatchNorm(4*N_filters, (3, 3), (1,1), merge5)

    merge6 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer5), layer2])
    layer6 = Conv2DReluBatchNorm(2*N_filters, (3, 3), (1,1), merge6)

    merge7 = Concatenate(axis=-1)([UpSampling2D(size=(2,2))(layer6), layer1])
    layer7 = Conv2DReluBatchNorm(N_filters, (3, 3), (1,1), merge7)

    output = Conv2D(1, (1, 1), strides=(1,1), activation='sigmoid', name='output')(layer7)    

    from keras.models import Model
    return Model(inputs=inputs, outputs=output)  




