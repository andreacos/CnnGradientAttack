
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


def show_figures(I, A, true_score, adv_score):

    """Displays input image, adversarial image, difference with scores in image labels

    """
    plt.figure(figsize=(8, 6), dpi=80)
    true_class = np.argmax(true_score)
    adv_class = np.argmax(adv_score)

    plt.subplot(1, 3, 1)
    plt.title('Original (class {}, score {:2.2f})'.format(true_class, true_score[true_class]))
    plt.imshow(I, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Adversarial (class {}, score {:2.2f})'.format(adv_class, adv_score[adv_class]))
    plt.imshow(A)
    plt.axis('off', cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Difference (false colors)')
    abs_difference = np.double(np.abs(A - I))
    plt.imshow(abs_difference, cmap=plt.get_cmap('Blues'))
    plt.axis('off')

    plt.show()
    return
