
"""
    2017-2018 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Benedetta Tondi (benedettatondi@gmail.com) and Andrea Costanzo (anreacos82@gmail.com)

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:
    B. Tondi, “Pixel-domain Adversarial Examples Against CNN-based Manipulation Detectors",
    Electronics Letters, 2018, DOI 10.1049/el.2018.6469
    (http://clem.dii.unisi.it/~vipp/files/publications/IETletter_CNNattacks_final.pdf)

"""

import os
import numpy as np
import matplotlib.pyplot as plt


def find_layer_idx(model, layer_name):
    """Looks up the layer index corresponding to `layer_name` from `model`.

    Args:
        model: The `keras.models.Model` instance.
        layer_name: The name of the layer to lookup.

    Returns:
        The layer index if found. Raises an exception otherwise.
    """
    layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            layer_idx = idx
            break

    if layer_idx is None:
        raise ValueError("No layer with name '{}' within the model".format(layer_name))
    return layer_idx


def force_linear_activation(model, layername='predictions', savemodel=None):

    """Switches activation's type to linear in the last layer

    """

    assert savemodel is None or isinstance(savemodel, str)

    from keras.models import load_model
    from keras import activations

    layer_idx = find_layer_idx(model, layername)
    model.layers[layer_idx].activation = activations.linear
    model.save('tmp.h5')
    # model.save('tmp.h5', include_optimizer=False)
    linmodel = load_model('tmp.h5')

    if savemodel is not None:
        os.rename('tmp.h5', savemodel)
    else:
        os.remove('tmp.h5')
    return linmodel


def softmax(x):

    """Transforms predictions into probability values.

    """

    assert x.ndim == 1
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def show_figures(img, adv, img_score, adv_score):

    assert img.shape == adv.shape

    display = show_figures_grayscale if len(img.shape) == 2 else show_figures_rgb
    return display(img, adv, img_score, adv_score)


def show_figures_rgb(I, A, true_score, adv_score):

    true_class = np.argmax(true_score)
    adv_class = np.argmax(adv_score)

    # Show input image
    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax0 = ax[0][0]
    ax0.imshow(I)
    ax0.axis('off')
    ax0.set_title('Original (class {}, score {:2.2f})'.format(true_class, true_score[true_class]))

    # Show attacked image
    ax0 = ax[0][1]
    ax0.imshow(A)
    ax0.axis('off')
    ax0.set_title('Adversarial (class {}, score {:2.2f})'.format(adv_class, adv_score[adv_class]))

    # Blank box
    ax[0][2].axis('off')

    # Show image differece channel-by-channel
    abs_difference = np.double(np.abs(A - I))
    cmaps = ['Reds', 'Greens', 'Blues']
    for i in range(0, 3):
        axi = ax[1][i]
        axi.set_title('Difference ({})'.format(cmaps[i][0:-1]))
        imi = axi.imshow(abs_difference[:, :, i], cmap=plt.get_cmap(cmaps[i]))
        fig.colorbar(imi, ax=axi, fraction=0.046, pad=0.04)
        axi.axis('off')

    return fig


def show_figures_grayscale(I, A, true_score, adv_score):

    """Displays input image, adversarial image, difference with scores in image labels

    """

    true_class = np.argmax(true_score)
    adv_class = np.argmax(adv_score)

    # Show input image
    fig, ax = plt.subplots(ncols=3)
    ax0 = ax[0]
    ax0.imshow(I)
    ax0.axis('off')
    ax0.set_title('Original (class {}, score {:2.2f})'.format(true_class, true_score[true_class]))

    # Show attacked image
    ax0 = ax[1]
    ax0.imshow(A)
    ax0.axis('off')
    ax0.set_title('Adversarial (class {}, score {:2.2f})'.format(adv_class, adv_score[adv_class]))

    # Show image difference channel-by-channel
    abs_difference = np.double(np.abs(A - I))
    ax2 = ax[2]
    im = ax2.imshow(abs_difference, cmap='Blues')
    fig.colorbar(im, ax=ax2, ticks=[0, np.max(abs_difference)], fraction=0.046, pad=0.04)
    ax2.axis('off')
    ax2.set_title('Difference (false colors)')

    return fig
