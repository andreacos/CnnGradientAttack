
import keras
from keras.models import load_model
from scipy.misc import imread
import os
from glob import glob
from PixelDomainAttack import *
from utils import force_linear_activation, show_figures, softmax

keras.backend.set_learning_phase(0)


if __name__ == '__main__':

    # Color script
    model_file = 'models/model_keras_ICIP18_64x64x3.h5'
    images_directory = '/media/D/Datasets/OriginalVSclahe_clip5_png/64x64/test/clahe/*.*'

    # True label, ground truth: we assume all images in directory belong to the same class! For the model we provide,
    # 0 = manipulated image and 1 = pristine image. Change this accordingly to your to-be-fooled model and input image!
    label = 0

    # Load Keras model and softmax with linear activations
    model = load_model(model_file)
    model = force_linear_activation(model=model, savemodel=None)

    # Retrieve information about CNN input size and output classes from model
    img_rows, img_cols, img_chans = model.layers[0].input_shape[1:]
    num_classes = model.layers[-1].output_shape[-1]

    # ------------------------------------------------------------------------------------------------------------------
    #  Load test data, preprocess (if CNN model requires it), define labels and test
    # ------------------------------------------------------------------------------------------------------------------

    images = glob(images_directory)
    numImg = len(images)
    print('Found {} images in directory {}. Assuming that all the images belong to the same class {}'
          .format(numImg,  images_directory, label))

    # Load all images into a "stack"
    x_test = np.zeros((numImg, img_rows, img_cols, img_chans))
    for i in np.arange(numImg):

        # Reshape test data, scale by 255 because the to-be-fooled CNN was trained with this preprocessing
        img = imread(images[i])
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        x_test[i] = img.astype('float32') / 255.0
    print('   > All images loaded.')

    # Create label array and convert to Keras' one-hot labels
    y_test_c = np.tile(label, numImg)
    y_test = keras.utils.to_categorical(y_test_c, num_classes)

    # Test legitimate examples
    print('Testing input images')
    score = model.evaluate(x_test, y_test, verbose=0)
    predicted_legitimate_labels = np.argmax(model.predict(x_test), axis=1)

    print('   > Accuracy on legitimate images (all): {:3.4f}'.format(score[1]))

    # ----------------------------------------------------------------------------------------------------------------------
    # Attack ***only for correctly classified examples***
    # ----------------------------------------------------------------------------------------------------------------------

    # Prepare attack
    print('WARNING: attack may take long or cause memory overflow if input image is too large')
    print('Attacking input images')
    gdm = PixelDomainAttackMethod(model=model, max_no_it=20, T=100, delta=1, k_stop=.8, k_increment=.002)

    # Extract correctly classified images and their labels
    correctly_classified = np.argwhere(predicted_legitimate_labels == y_test_c).squeeze()
    n_test = len(correctly_classified) 
    test_images = np.uint8(x_test[correctly_classified[0:n_test], :, :, :] * 255).astype('float32')
    true_labels_cat = y_test[correctly_classified[0:n_test], :]
       
    # Compute adversarial examples for n_test images (correctly classified by the Net)
    print('   > Excluding {} images that were not classified correctly (no need to attack)'
          .format(numImg - len(correctly_classified)))

    # Initialize counters & containers
    failures = avg_Max_dist = avg_L1_dist = avg_No_Mod_Pixels = 0
    adv_images = np.zeros(test_images.shape)

    # Start attacking each image of the stack
    for idx in np.arange(n_test):

        filename = images[correctly_classified[idx]]
        print('   > Image {} - {} of {}'.format(os.path.basename(filename), idx+1, n_test))

        # Generate adversarial image
        adv_images[idx], _, _, it_number = gdm.generate_attack(x=test_images[idx])

        # Score for current legitimate image
        true_score, true_class = keras_predictor(model, test_images[idx])
        true_score = true_score[0]
        true_score_soft = softmax(true_score)

        # Score for the corresponding adversarial image
        adv_score, adv_class = keras_predictor(model, adv_images[idx])
        adv_score = adv_score[0]
        adv_score_soft = softmax(adv_score)

        print('      > Class changed from {} to {} after {} iterations'.format(true_class, adv_class, it_number))
        print('      > Score changed from {} to {}'.format(true_score_soft, adv_score_soft))

        diff_matrix = adv_images[idx] - test_images[idx]

        print('      > Max distortion = {:3.4f}; L1 distortion = {:3.4f}'
              .format(abs(diff_matrix).max(), abs(diff_matrix).sum() / test_images[0].size))
        print('      > % of modified pixels on integers = {:3.4f}. % of negative modifications = {:3.4f}'
                .format(np.count_nonzero(diff_matrix) / test_images[0].size,
                        np.count_nonzero(np.double(abs(diff_matrix)) - np.double(diff_matrix)) /
                            (img_rows * img_cols*img_chans)))

        # WARNING! Plot each image, adversarial_image and difference (test_images are in [0,255])
        # show_figures(test_images[idx].astype('uint8').squeeze(),
        #              adv_images[idx].astype('uint8').squeeze(), true_score_soft, adv_score_soft)

        # Update average distortion
        if true_class != adv_class:
            avg_Max_dist += abs(diff_matrix).max()
            avg_L1_dist += abs(diff_matrix).sum() / test_images[0].size
            avg_No_Mod_Pixels += np.count_nonzero(diff_matrix) / test_images[0].size
        else:
            failures += 1

    # Evaluate accuracy
    success = n_test - failures
    true_labels_cat = np.array(true_labels_cat)
    adv_score = model.evaluate(adv_images/255.0, true_labels_cat, verbose=0)

    def report():
        print('Attack failed {} times out of {}'.format(failures, n_test))
        print('Average distortion: max dist {}, L1 dist {}'.format(avg_Max_dist / success, avg_L1_dist / success))
        print('Average no of modified pixels: {}'.format(avg_No_Mod_Pixels / success))
        print('Accuracy on legitimate images (all): {:3.4f}'.format(score[1]))
        print('Accuracy on adversarial images: {:3.4f}'.format(adv_score[1]))
        return

    report()
