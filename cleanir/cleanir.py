import os
import numpy as np
from keras.models import load_model


class Cleanir:
    """CLEANIR class
    """
    def __init__(self, dsize=(64, 64), latent_dim=512):
        self.__encoder = None
        self.__decoder = None
        self.__dn = None

        self.__dsize = dsize
        self.__latent_dim = latent_dim

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
