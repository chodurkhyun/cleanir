import os
import numpy as np
from keras.models import load_model
from cleanir.blocks import *
from cleanir.losses import *
from cleanir.tools.named_logs import named_logs

from keras.layers import Input, AveragePooling2D, Flatten, Dense, concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from tqdm.autonotebook import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import face_recognition


class Cleanir:
    """CLEANIR class
    """
    def __init__(self, dsize=(64, 64), latent_dim=512):
        self.__encoder = None
        self.__decoder = None
        self.__dn = None
        self.__cleanir_net = None

        self.__dsize = dsize
        self.__latent_dim = latent_dim
        self.__input_shape = (*dsize, 3)

    def load_models(self, model_path):
        """loads models for CLEANIR from model files

        Arguments:
            model_path {string} -- model folder path

        Returns:
            boolean -- whether loading the models is successful or not
        """
        if not os.path.isdir(model_path):
            print('ERROR: model_path might be invalid or not exist')
            return False

        encoder_path = os.path.join(model_path, 'encoder.h5')
        decoder_path = os.path.join(model_path, 'decoder.h5')
        dn_path = os.path.join(model_path, 'dn.h5')

        if not (os.path.exists(encoder_path) and
                os.path.exists(decoder_path) and
                os.path.exists(dn_path)):
            print('ERROR: encoder.h5 or decoder.h5 or dn.h5 does not exist')
            return False

        self.__encoder = load_model(encoder_path, compile=False)
        self.__decoder = load_model(decoder_path, compile=False)
        self.__dn = load_model(dn_path, compile=False)
        return True

    def encode(self, face_img):
        """encodes a face image to latent vectors (zi, za).

        Arguments:
            face_img {np.array} -- a cropped face image

        Returns:
            tuple -- (zi, za)
        """
        normalized = (face_img / 127.5 - 1).reshape((1, *self.__dsize, 3))
        z = self.__encoder.predict(normalized)[2]
        zi, za = self.__dn.predict(z)
        return zi, za

    def manipulate(self, zi, degree, z90=None):
        """manipulates an identity vector (zi) and returns the manipulated identity vector.

        Arguments:
            zi {tensor} -- an identity vector
            degree {integer} -- degrees of manipulation

        Keyword Arguments:
            z90 {tensor} -- a manipulated identity vector by 90-degrees,
                            a rotational axis is set by assigning this.
                            if it is None, the axis will be set randomly
                            (default: {None})

        Returns:
            tensor -- manipulated identity vector
        """

        if z90 is None:
            # gram-schmidt process
            zr = np.random.uniform(-1., 1., (1, self.__latent_dim))
            zr /= np.linalg.norm(zr)
            z90 = zr - np.matmul(zi, zr.T) * zi

        radian = np.deg2rad(degree)
        return zi * np.cos(radian) + z90 * np.sin(radian)

    def decode(self, zi, za):
        """decodes latent vectors (zi, za) to generate a face image.

        Arguments:
            zi {tensor} -- an identity vector
            za {tensor} -- an attribute vector

        Returns:
            np.array -- a generated face image
        """
        generated = ((self.__decoder.predict([zi, za]) + 1.) * 127.5).astype('uint8')
        return generated.reshape(*self.__dsize, 3)

    def deidentify(self, face_img, degree):
        """deidentifies an face image by the input degrees.

        Arguments:
            face_img {np.array} -- an cropped face image
            degree {integer} -- degrees of manipulation

        Returns:
            np.array -- a generated face image
        """
        zi, za = self.encode(face_img)
        zm = self.manipulate(zi, degree)
        return self.decode(zm, za)

    def get_deid_single_axis_func(self, face_img):
        """get a deidentification along a single axis function.

        Arguments:
            face_img {np.array} -- an cropped face image

        Returns:
            function -- deid function using the input face
        """
        zi, za = self.encode(face_img)
        z90 = self.manipulate(zi, 90)

        def deid_single_axis(degree):
            zm = self.manipulate(zi, degree, z90)
            return self.decode(zm, za)

        return deid_single_axis

    def build_network(self, n_blocks=4, recon_weight=0.3, with_landmark_loss=False):
        """Build CLEANIR network 

        Args:
            n_blocks (int, optional): [the number of encoder/decoder blocks]. Defaults to 4.
            recon_weight (float, optional): [reconstruction loss weight of VAE loss]. Defaults to 0.3.
            with_landmark_loss (bool, optional): [whether to use landmark loss]. Defaults to False.
        """
        units = 512
        start_units = 64
        decoder_start_w = 4

        # Encoder
        input_x = Input(self.__input_shape)
        h = input_x

        h = Conv2D(64, 7, strides=(2, 2), padding='same')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = LeakyReLU(0.2)(h)

        for i in range(1, n_blocks + 1):
            h = resnet_conv_block(h, i * start_units, i + 1, 'conv')
            h = resnet_identity_block(h, i * start_units, i + 1, 'identity')

        h = AveragePooling2D((3, 3), padding='same')(h)
        h = Flatten()(h)

        for _ in range(1):
            h = Dense(self.__latent_dim)(h)
            h = LeakyReLU(0.2)(h)
            h = BatchNormalization(momentum=0.8)(h)

        mean = Dense(self.__latent_dim, name='encoder_mean')(h)
        logvar = Dense(self.__latent_dim, name='encoder_sigma')(h)
        z = Lambda(sampling, output_shape=(self.__latent_dim, ))([mean, logvar])

        encoder = Model(input_x, [mean, logvar, z], name='Encoder')

        # disentangle network
        input_z = Input(shape=(self.__latent_dim, ))
        h = input_z

        half_dim = self.__latent_dim // 2
        z1 = Lambda(lambda x: x[:, :half_dim])(h)
        z1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(z1)
        z2 = Lambda(lambda x: x[:, half_dim:])(h)

        dn = Model(input_z, [z1, z2], name='DisentangleNet')

        # Decoder
        input_z1 = Input(shape=(self.__latent_dim // 2, ))
        input_z2 = Input(shape=(self.__latent_dim // 2, ))
        h = concatenate([input_z1, input_z2])
        h = Dense(decoder_start_w * decoder_start_w * units)(h)
        h = Reshape((decoder_start_w, decoder_start_w, units))(h)  # (4, 4, 256)
        h = LeakyReLU(0.2)(h)

        for i in range(1, n_blocks + 1):
            h = resnet_conv_block(h, units // 2 ** i, i + 1, 'conv_dec', upscale=True)
            h = resnet_identity_block(h, units // 2 ** i, i + 1, 'identity_dec')

        h = Conv2D(3, (7, 7), padding='same', activation='tanh')(h)

        decoder = Model([input_z1, input_z2], h, name='Decoder')

        zi, za = dn(encoder(input_x)[2])
        x_tilde = decoder([zi, za])

        input_true_embedding = Input(shape=(self.__latent_dim // 2, ))

        cleanir_net = Model([input_x, input_true_embedding],
                            [x_tilde, zi, za],
                            name='Cleanir_Net')

        vae_losses = vae_loss((input_x + 1.) / 2.,
                              (x_tilde + 1.) / 2.,
                              mean, logvar, recon_weight=recon_weight)

        em_loss = 1000 * K.sum(1. - K.sum(input_true_embedding * zi, axis=-1))

        landmark_loss = None
        lo = None
        lg = None
        if with_landmark_loss:
            lo = extract_facial_landmarks(input_x)
            lg = extract_facial_landmarks(x_tilde)
            landmark_loss = K.sum(K.square(lo - lg), axis=-1)
        

        cleanir_net.add_loss(vae_losses)
        cleanir_net.add_loss(em_loss)

        if with_landmark_loss:
            cleanir_net.add_loss(landmark_loss)

        adam = Adam(lr=1e-4, decay=1e-6)
        cleanir_net.compile(optimizer=adam)

        cleanir_net.metrics_tensors.append(vae_losses)
        cleanir_net.metrics_names.append('vae_loss')

        cleanir_net.metrics_tensors.append(em_loss)
        cleanir_net.metrics_names.append('em_loss')

        if with_landmark_loss:
            cleanir_net.metrics_tensors.append(landmark_loss)
            cleanir_net.metrics_names.append('landmark_loss')

        self.__encoder = encoder
        self.__decoder = decoder
        self.__dn = dn
        self.__cleanir_net = cleanir_net

        plot_model(self.__encoder, show_shapes=True, to_file='encoder.png')
        plot_model(self.__decoder, show_shapes=True, to_file='decoder.png')
        plot_model(self.__dn, show_shapes=True, to_file='dn.png')
        plot_model(self.__cleanir_net, show_shapes=True, to_file='cleanir_net.png')
    
    def print_network_summary(self):
        """print summary of networks
        """
        if self.__encoder:
            self.__encoder.summary()
        
        if self.__decoder:
            self.__decoder.summary()
        
        if self.__dn:
            self.__dn.summary()
        
        if self.__cleanir_net:
            self.__cleanir_net.summary()

    def train(self, res_path, data_generator, n_epochs=300):
        """train the network

        Args:
            res_path ([str]): [path for saving the results]
            data_generator ([func]): [data generator for the training set]
            n_epochs (int, optional): [the number of epochs]. Defaults to 300.
        """
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        X_test, c_test = data_generator[0]
        loss = None

        tb_callback = TensorBoard(log_dir='{0}/Graph'.format(res_path),
                                  histogram_freq=0, write_graph=True,
                                  write_images=True)
        tb_callback.set_model(self.__cleanir_net)

        batch_step = 0
        for epoch in range(1, n_epochs + 1):
            for X, c in tqdm(data_generator):
                loss = self.__cleanir_net.train_on_batch([X, c], None)

                tb_callback.on_epoch_end(batch_step, named_logs(self.__cleanir_net, loss))
                batch_step += 1

            print("epoch", epoch, "loss", loss)

            data_generator.on_epoch_end()

            if epoch % 5 == 0 or epoch == 1:
                self.__encoder.save('{0}/encoder{1}.h5'.format(res_path, epoch))
                self.__decoder.save('{0}/decoder{1}.h5'.format(res_path, epoch))
                self.__dn.save('{0}/dn{1}.h5'.format(res_path, epoch))
                self.__cleanir_net.save('{0}/cleanir_net{1}.h5'.format(res_path, epoch))

                res = self.__cleanir_net.predict([X_test, c_test])

                plt.figure(figsize=(15, 15))
                idx = 1
                for i in range(0, 30, 3):
                    plt.subplot(10, 10, idx)
                    plt.imshow((X_test[i] + 1) / 2)
                    idx += 1

                    x = res[0][i].reshape((1, 64, 64, 3))
                    plt.subplot(10, 10, idx)
                    plt.imshow(((x + 1) / 2).reshape((64, 64, 3)))
                    idx += 1

                    for j in range(8):
                        x_pred = self.__decoder.predict([res[1][j].reshape((1, -1)),
                                                        res[2][i].reshape((1, -1))])

                        plt.subplot(10, 10, idx)
                        plt.imshow(((x_pred + 1) / 2).reshape((64, 64, 3)))
                        idx += 1

                plt.savefig('{0}/res__{1}.png'.format(res_path, epoch))
                plt.close()

        tb_callback.on_train_end(None)
