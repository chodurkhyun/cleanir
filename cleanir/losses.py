from keras import backend as K
from keras.losses import binary_crossentropy


def kl_loss(y_true, y_pred):
    """KL Loss of VAE

    Arguments:
        y_true {tensor} -- not needed
        y_pred {tensor} -- z_mean and z_logvar
    """

    z_dim = y_pred.shape[-1] // 2
    z_mean = y_pred[:z_dim]
    z_logvar = y_pred[z_dim:]

    return -0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1)


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return binary_crossentropy(K.flatten((y_true + 1.) / 2.),
                               K.flatten((y_pred + 1.) / 2.))


def vae_loss(y_true, y_pred, z_mean, z_logvar, flatten=False, recon_loss_type='bce',
             recon_weight=0.5):
    """
    Calculate the vae loss.

    :param y_true: input
    :param y_pred: decoded
    :param z_mean: mean of Q(z|X), outputs of the encoder
    :param z_logvar: log of variance of Q(z|X), outputs of the encoder
    :param recon_loss_type: 'bce' or 'mse'
    :return: vae loss
    """

    kl_loss = -0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1)

    if recon_loss_type == 'bce':
        if flatten:
            bce_loss = binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred)) * int(np.prod(y_true.shape[1:]))
        else:
            bce_loss = K.sum(binary_crossentropy(y_true, y_pred)) * int(y_true.shape[-1])

        return K.mean(recon_weight * bce_loss + (1. - recon_weight) * kl_loss)

    elif recon_loss_type == 'mse':
        reconstruction_loss = mse(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        total_loss = reconstruction_loss + 0.0001 * kl_loss

        return total_loss