from scipy.misc import imread
from utils import *
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    A = imread('/media/D/Datasets/OriginalVSclahe_clip5_png/64x64/test/clahe/rf4a7056bt.TIF_crop0.png')
   # B = imread('/media/D/Datasets/OriginalVSclahe_clip5_png/64x64/test/clahe/rf4a7056bt.TIF_crop30.png')

    #plt.show(show_figures(A,B,[0,1],[1,0]))

    f = '/media/D/Datasets/RAISE8Ksplit/test/rdc85e63bt.TIF'
    img = cv2.imread(f)
    size_image = 64
    image_channels = 3

    orig = img

    # Convert image to the HSV color space and perform CLAHE on V channel
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    cl = clahe.apply(v)

    # Back to BGR
    v_img = cv2.merge((h, s, cl))
    img = cv2.cvtColor(v_img, cv2.COLOR_HSV2BGR)

    multiple_height = int(np.floor(img.shape[0] / float(size_image)) * size_image)
    multiple_width = int(np.floor(img.shape[1] / float(size_image)) * size_image)

    count = 0
    img = img[:multiple_height, :multiple_width, :image_channels]
    for k in range(0, multiple_height, size_image):
        for l in range(0, multiple_width, size_image):
            crop_img = orig[k:k + size_image, l:l + size_image, :image_channels]
            crop_img_eq = img[k:k + size_image, l:l + size_image, :image_channels]

            cv2.imwrite(os.path.join('temp',
                                    '{}_crop{}.png'.format(os.path.basename(f), count)),
                                    crop_img_eq, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            count += 1