from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, ReLU, \
                         BatchNormalization, Add, UpSampling2D, Lambda
from keras.regularizers import l2
from keras.initializers import glorot_uniform
import face_recognition
from keras import backend as K

def extract_facial_landmarks(img):
    img = (img + 1.) * 127.5
    landmarks_list = face_recognition.face_landmarks(img)
    landmarks = []
    for _, v in landmarks_list[0].items():
        landmarks += [*v]
    landmark_ret += [landmarks]
    
    return landmark_ret

def facial_landmark_block(X):
    X = Lambda(extract_facial_landmarks)(X)
    return X

def sampling(args):
    """ Reparameterization trick

    :param args: (mean, log of variance) of Q(z|X)
    :return: sampled latent vector
    """

    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    number_of_zdim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch_size, number_of_zdim))

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def resnet_identity_block(X, filters, stage, block, kernel_size=(3, 3)):
    conv_name_base = 'res{0}{1}_branch'.format(stage, block)
    bn_name_base = 'bn{0}{1}_branch'.format(stage, block)

    X_shortcut = X

    for i in ['2a', '2b']:
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                   name=conv_name_base + i, padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(name=bn_name_base + i)(X)
        X = LeakyReLU(alpha=0.2)(X)

    X = Add()([X, X_shortcut])
    X = LeakyReLU(alpha=0.2)(X)

    return X


def resnet_conv_block(X, filters, stage, block, kernel_size=(3, 3),
                      upscale=False):
    conv_name_base = 'res{0}{1}_branch'.format(stage, block)
    bn_name_base = 'bn{0}{1}_branch'.format(stage, block)

    X_shortcut = X

    if upscale:
        X = UpSampling2D()(X)
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                   name=conv_name_base + '2a', padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(name=bn_name_base + '2a')(X)
        X = LeakyReLU(alpha=0.2)(X)

    else:
        X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(2, 2),
                   name=conv_name_base + '2a', padding='same',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(name=bn_name_base + '2a')(X)
        X = LeakyReLU(alpha=0.2)(X)

    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
               name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = LeakyReLU(alpha=0.2)(X)

    if upscale:
        X_shortcut = UpSampling2D()(X_shortcut)
        X_shortcut = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                            name=conv_name_base + '1', padding='same',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)
    else:
        X_shortcut = Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=(2, 2),
                            name=conv_name_base + '1', padding='same',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = LeakyReLU(alpha=0.2)(X)

    return X


def conv_block(x,
               filters,
               leaky=True,
               leaky_relu_alpha=0.2,
               wdecay=1e-5,
               bn_mom=0.9,
               bn_eps=1e-6,
               name=''):
    layers = [
        Conv2D(filters, 5, strides=2, padding='same',
               kernel_regularizer=l2(wdecay),
               kernel_initializer='he_uniform', name=name + 'conv'),
        BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
        LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
    ]

    if x is None:
        return layers

    for layer in layers:
        x = layer(x)
    return x


def deconv_block(x,
                 filters,
                 leaky=True,
                 leaky_relu_alpha=0.2,
                 wdecay=1e-5,
                 bn_mom=0.9,
                 bn_eps=1e-6,
                 name=''):
    layers = [
        Conv2DTranspose(filters, 5, strides=2, padding='same',
                        kernel_regularizer=l2(wdecay),
                        kernel_initializer='he_uniform', name=name + 'conv'),
        BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
        LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
    ]

    if x is None:
        return layers

    for layer in layers:
        x = layer(x)
    return x

