
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

import keras
from keras.models import load_model
from scipy.misc import imread
from PixelDomainAttack import *
from utils import force_linear_activation, show_figures, softmax

keras.backend.set_learning_phase(0)


if __name__ == '__main__':

    # Grayscale demo
    # model_file = 'models/<your_grayscale_keras_model.h5>'
    # img_file = 'resources/<your_grayscale_test_image>'

    # Color demo
    model_file = '/media/D/Andrea/Models/model_keras_ICIP18_64x64x3.h5'
    img_file = 'resources/sample_color_clahe.png'

    # Load Keras model and softmax with linear activations
    model = load_model(model_file)
    model = force_linear_activation(model=model, savemodel=None)

    # Load image: for single-channel images we need a shape = (rows, cols, 1) rather than (rows, cols)
    img = imread(img_file)
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    # True label: this is the ground truth! For the model we provide, 0 = manipulated image and 1 = pristine image.
    # Change this accordingly to the to-be-fooled model and input image!
    true_class = 0

    assert img.shape == model.layers[0].input_shape[1:]

    # Reshape input image as (1, rows, cols, channels), then divide by 255 (our nets were trained this way)
    pred_score = model.predict(np.expand_dims(img.astype('float32'), axis=0) / 255, verbose=0)
    pred_class = np.argmax(pred_score)

    if pred_class != true_class:

        print('Input image is already miss-classified: there is no need to attack')
    else:

        # Attack
        gdm = PixelDomainAttackMethod(model=model, delta=1, max_no_it=20, T=100, k_stop=.8, k_increment=.002)
        adv_img, _, _, it_number = gdm.generate_attack(x=img)

        adv_score = model.predict(np.expand_dims(adv_img.astype('float32'), axis=0)/255.)
        adv_class = np.argmax(adv_score)

        diff_matrix = adv_img - img

        def report():
            print('Attack ended after {} iterations'.format(it_number))
            print('Class changed from {} to {}'.format(true_class, adv_class))
            print('Score changed from {} to {}'.format(softmax(pred_score[0]), softmax(adv_score[0])))
            print('Max distortion = {:3.4f}'.format(abs(diff_matrix).max()))
            print('L1 distortion = {:3.4f}'.format(abs(diff_matrix).sum() / img.size))
            print('Percentage of modified pixels on integers = {:3.4f}'
                  .format(np.count_nonzero(diff_matrix) / img.size))

            # Plot image, adv_image and difference
            show_figures(img.squeeze(), adv_img.squeeze(), softmax(pred_score[0]), softmax(adv_score[0]))

            return

        report()
