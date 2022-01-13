import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
@author: Ugur AKYEL
@st id: 20190808020
"""


def prepare_rgb2gray(img_location):
    """
    This method only prepare the images cv2 library and
    rgb to Gray image
    Because SIFT methos uses the Grayscale image formation.
    """
    img = cv2.imread(img_location)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sift_compare_mask(mask_no, train_img_no):
    """
        Firstly we prepare our images for detect the keypoints and desriptor.
        And we convert both of them to Grayscale format our SIFT object
        then we create a BruteForce Matcher object for the Decriptors
        and also produce a fetured comparing image
    :param mask_no: mask image for detection
    :param train_img_no: train images for searching our dataset
    :return: an image list for one result image fully compare only
    """
    img3 = []
    result = []
    mask_img = prepare_rgb2gray('masks/cvmask{0}.jpg'.format(mask_no))
    train_img = prepare_rgb2gray('mask_dataset/maksssksksss{0}.png'.format(train_img_no))

    sift = cv2.SIFT_create()  #creating SIFT object

    keypoints_1, descriptors_1 = sift.detectAndCompute(mask_img, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(train_img, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)  # we use NORML! for SIFT because most efficient brute for
    # matches trecnique for us

    if descriptors_1 is not None and descriptors_2 is not None:
        # If the descriptor is NoneType tha will create an
        # error for our program
        matches = bf.match(descriptors_1, descriptors_2)  # BruteForce Matcher for nonempty descriptors

        matches = sorted(matches, key=lambda x: x.distance) #sorting process for matches tuple

        img3 = cv2.drawMatches(mask_img, keypoints_1, train_img, keypoints_2, #drawing process for two images
                               matches, train_img, flags=2)
        result.append(img3)  #appending the our table

    print('images: ', mask_no, train_img_no, 'length: ', len(result))
    return img3


if __name__ == '__main__':
    #we create a 3d matrix for our result output matches images
    x = [[sift_compare_mask(i, j) for i in range(13)] for j in range(149)]
    print("x's shape: ", len(x), len(x[0]))

    for h in range(149):  # for 149 serachinf images for SIFT
        for w in range(13):   # we compare and detect feature with this 13 images
            plt.imshow(x[h][w], cmap='gray')
            plt.title('Mask:' + str(w) + '-train:' + str(h), fontsize=7)
            plt.savefig("output_images/train" + str(h) + "/output" + str(w) + "-complete.png")
