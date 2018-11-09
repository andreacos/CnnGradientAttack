
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

import numpy as np


def keras_predictor(model, x):

    X = np.reshape(x/255, (1,) + x.shape) if len(x.shape) == 3 else x/255

    class_score = np.array(model.predict(X, verbose=0))
    class_label = np.argmax(class_score, axis=1)
    return class_score, class_label


def keras_predictor_distilledSoftmax(model, x, T):

    # param T: normalization parameter (distillation)

    X = np.reshape(x/255, (1,) + x.shape) if len(x.shape) == 3 else x/255

    class_score_lin = np.array(model.predict(X, verbose=0))
    class_label = np.argmax(class_score_lin, axis=1)

    assert class_score_lin.ndim == 2

    class_score = np.zeros(class_score_lin.shape)
    for idx in np.arange(class_score_lin.shape[0]):
        class_score[idx] = np.exp(class_score_lin[idx]/T)
        class_score[idx] = class_score[idx] / np.sum(class_score[idx])

    return class_score, class_label


class PixelDomainAttackMethod:

    """
    Gradient-inspired pixel-domain attack against CNN detection of global image manipulations
    (Pdf link: http://clem.dii.unisi.it/~vipp/files/publications/PID5052975.pdf)
    Please cite: B.Tondi. 'Pixel-domain adversarial examples against CNN-based manipulation detectors', Electronic Letters, Vol. 54, no. 21, October 2018, p. 1220 – 1222
    """

    def __init__(self, model=None, delta=1, max_no_it=20, T=100, k_stop=.8, k_increment=.002):
        """
        Create a PixelDomainAttackMethod instance.

        :param model: model to-be-fooled CNN model
        :param delta: small increment applied to a pixel position (the default value is 1)
        :param max_no_it: maximum number of iterations of the attack
        :param T: distillation parameter for the softmax during the gradient computation (default value 100)
               T = 'none' : no distillation (standard softmax)
        :param k_stop: upper bound of the K search range
        :param k_increment: search step
        :return:
        """

        self._model = model
        self._delta = delta
        self._max_no_it = max_no_it
        self._T = T
        self._k_stop = k_stop
        self._k_increment = k_increment
        self._image = None
        self._truelabel = None

    def generate_attack(self, x, x_label=None):
        """
        Gradient-based pixel-domain attack against CNN detection of global image manipulations
        Paper link: http://clem.dii.unisi.it/~vipp/files/publications/PID5052975.pdf
        :param x: input image
        :param x_label: class label of the input image. If None, label is "guessed" by the attack
        :return: Adversarial image
        """

        def compute_gradient(img, label, model, delta, gI, T):
            """
            Compute a rough approximation of the output gradient wrt input image which fits the integer nature
            of the image pixels
            :param img: input image
            :param label: class label of the input image
            :param model: to-be-fooled CNN model (Keras)
            :param delta: increment applied to a pixel position
            :param gI: classification score of the input image
            :param T: distillation parameter for the softmax
            :return: Gradient approximation
            """

            n, m = img.shape[0], img.shape[1]

            # Define vector of all the modified images
            reshaped_img = np.expand_dims(img, 0)

            y = np.tile(reshaped_img, (img.shape[-1]*(n*m), 1, 1, 1)) # the image is assumed 3 channels
            gIDeltaMatrix = np.zeros(img.shape)

            # Create an [nm x nm] identity matrix with a delta value on the pixel that must be incremented
            delta_ij = delta * np.eye(n * m, n * m)

            for idx in np.arange(img.shape[2]):

                # Reshape the [n x m] image into a [1 x nm] array, then replicate nxm times over the rows
                res = np.reshape(img[:, :, idx], (1, n*m))
                img_reshaped = np.tile(res, (n*m, 1))

                # Add delta to the reshape input, then reshape again to a [n x m x nm] matrix where channel i-th
                # corresponds to the input image with pixel i-th incremented by delta (pixels j for j != i unchanged)
                y[idx*(n*m):idx*(n*m) + n*m, :, :, idx] = np.reshape((img_reshaped + delta_ij), (n*m, n, m))
                s = y[idx*(n*m):idx*(n*m) + n*m, :, :, :]

                # Clipping up to 255 and below 0
                y[y > 255] = 255
                y[y < 0] = 0

                # Predict class of each version of the incremented image and then reshape back to original input size
                if T == 'none':
                    gIDelta, _ = keras_predictor(model, s)

                else:
                    # Use distillation to increase the sensitivity of the softmax output to a pixel change in the input
                    gIDelta, _ = keras_predictor_distilledSoftmax(model, s, T)

                gIDelta = np.reshape(gIDelta[:, label], (n, m))
                gIDeltaMatrix[:, :, idx] = gIDelta

            # Return gradient
            return ((gI - gIDeltaMatrix) / self._delta).squeeze()

        def algorithm_search_best_k(img, label, model, gradient, delta, step=.002, k_min=.8):
            """
            Refine search for best attack parameters.
            :param img: input to-be-attacked image
            :param label: class label of the input image
            :param model: to-be-fooled cnn model
            :param gradient: the rough approximation of the output gradient wrt input image
            :param delta: increment applied to a pixel position
            :param step: search step for best K
            :param k_min: minimum allowed value for K
            :return:
            """
            k = 1 - step

            # Keep going until one of the two conditions becomes true: either the minimum allowed K is reached
            # or attack was successful. In the latter case, start bisection to refine the attack
            k_condition, s_condition = False, False

            while not k_condition and not s_condition:

                # Compute gradient's normalized histogram
                histo_grad, histo_bins = np.histogram(np.abs(gradient), bins='auto')
                histo_grad = histo_grad / np.sum(histo_grad, axis=None)

                # ((1-k)*100)% of the entries of the gradient matrix have value smaller than histo_bins[i_h]
                h_cumsum = np.cumsum(histo_grad)
                i_h = np.argwhere(h_cumsum > k)[0]

                if histo_bins[i_h] != 0:
                    epsilon = .5 / histo_bins[i_h]
                else:
                    epsilon = .5 / np.min(histo_grad[np.argwhere(histo_grad != 0)])

                delta_ij = np.round(epsilon * gradient)
                delta_ij[delta_ij > 1] = delta
                delta_ij[delta_ij < -1] = - delta
                delta_ij = np.reshape(delta_ij, img.shape)

                x_star = img + delta_ij
                x_star[x_star > 255] = 255
                x_star[x_star < 0] = 0
                delta_new_ij = x_star - img
                delta_ij_matrix_trunc = np.reshape(delta_new_ij, img.shape)

                # Update K and check condition
                k -= step
                if k < k_min:
                    k_condition = True

                # Test the class_label and check if it is changed
                c_score, c_label = keras_predictor(model, x_star)

                if c_label != label:
                    s_condition = True

            return x_star, k+step, delta_ij_matrix_trunc

        # Keep track of the input image and its modifications
        self._image = x

        # Determine class score and label of the input image (if no label has been provided as input)
        input_score, guess_label = keras_predictor(self._model, self._image)

        self._truelabel = int(x_label) if x_label is not None else int(guess_label)

        # --------------------------------------------------------------------------------------------------------------
        # Main loop of algorithm (1)
        # --------------------------------------------------------------------------------------------------------------

        # Attack goes on until the input image changes class
        no_it = 0
        delta_overall_mod = np.zeros(self._image.shape)
        delta_abs_overall = np.zeros(self._image.shape)

        print('Starting attack iterations:')
        iter_label = self._truelabel
        while iter_label == self._truelabel:

            # Count number of iteration of the while
            no_it += 1
            print(' > Iteration #{}.. '.format(no_it))

            if no_it > self._max_no_it:
                print('Reached maximum number of iterations: {}. Giving up.'.format(self._max_no_it))
                break

            # Estimate gradient
            grad = compute_gradient(self._image, self._truelabel, self._model, self._delta,
                                    input_score[0][self._truelabel], self._T)

            if grad.max() == grad.min():  # if grad.max() == 0 & grad.min() == 0:
                print('Zero gradient! The image cannot be attacked (Hint: changing parameter T may help)')
                break

            # Attack and Search for best K
            adv_img, k, delta_matrix = algorithm_search_best_k(img=self._image,
                                                               label=self._truelabel,
                                                               model=self._model,
                                                               gradient=grad,
                                                               delta=self._delta,
                                                               step= self._k_increment,
                                                               k_min=self._k_stop)

            # Global matrix of modification
            delta_overall_mod = delta_overall_mod + delta_matrix
            delta_abs_overall = delta_abs_overall + abs(delta_matrix)

            # Update adversarial image I* and check its class
            self._image = adv_img
            score, iter_label = keras_predictor(self._model, self._image)

        return np.uint8(self._image), delta_overall_mod, delta_abs_overall, no_it
