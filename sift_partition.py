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
            and we divide an train image for 100x100 piksel format and each partition
            comparing with the mask images for obtain the SIFT
            detector finally we obtain produced a featured comparing image list
            for one train image and 13 mask images set
        :param mask_no: mask image for detection
        :param train_img_no: train images for searching our dataset
        :return: an image list for one result images list 1d
        """
    result = []
    mask_img = prepare_rgb2gray('masks/cvmask{0}.jpg'.format(mask_no))
    train_img = prepare_rgb2gray('mask_dataset/maksssksksss{0}.png'.format(train_img_no))

    sift = cv2.SIFT_create()
    print(train_img.shape)
    h, w = train_img.shape
    for i in range(0, h, 100): #for the divided by 100x100
        for j in range(0, w, 100):
            if i + 100 < h and j + 100 < w:
                """
                h: height for train image
                w: width for train image
                This for in for loop statement is require for providing image boundary
                And also i mwant to show local image region for each image; created 80x80 matrix
                for each searching image, Because Sift methor local feature detection is
                 not suitable with this scehma for mask detection. 
                 And also i choose only which region has a lot of descrptor matches 
                """
                print('i: {0} j: {1}'.format(i, j))
                keypoints_1, descriptors_1 = sift.detectAndCompute(mask_img, None)
                keypoints_2, descriptors_2 = sift.detectAndCompute(train_img[i:i + 100, j:j + 100], None)
                print(type(descriptors_2))
                print("Mask`s keypoint numbers: {0}, Train images keypoint "
                      "numbers {1}".format(len(keypoints_1), len(keypoints_2)))

                bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                min_matches = 0
                max_matches = ()
                # If the descriptor is NoneType tha will create an error for our program
                if descriptors_1 is not None and descriptors_2 is not None:
                    matches = bf.match(descriptors_1, descriptors_2)
                    print(len(matches), matches)
                    if len(matches) > min_matches: #we provide for the good image featured
                        max_matches = matches
                if max_matches != ():
                    max_matches = sorted(max_matches, key=lambda x: x.distance)

                    img3 = cv2.drawMatches(mask_img, keypoints_1, train_img[i:i + 100, j:j + 100], keypoints_2,
                                           max_matches, train_img[i:i + 100, j:j + 100], flags=2)
                    result.append(img3)

    print('images: ', mask_no, train_img_no, 'length: ', len(result))
    return result


"""
        plt.figure()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
        search_params = dict(checks=100)

        # Create the Flann Matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Obtain matches using K-Nearest Neighbor Method
        # the result 'matchs' is the number of similar matches found in both images
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Store good matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        img4 = cv2.drawMatches(mask_img, keypoints_1, train_img[i:i + 50], keypoints_2, good_matches, train_img, flags=2)
        plt.imshow(img4), plt.show()
"""

if __name__ == '__main__':
    x = [[sift_compare_mask(i, j) for i in range(13)] for j in range(149)]
    print("x's shape: ", len(x), len(x[0]), len(x[0][0]))
    """
        We want to obtain 3d image table row and column is mask images 
        and train images and each cell has image list for the matches 
    """
    for h in range(149):
        for w in range(13):
            fig, ax = plt.subplots(3, 5)
            k = 0
            for j in range(3):
                if k >= len(x[h][w]) or k >= 15:
                    break
                for i in range(5):
                    if k >= len(x[h][w]) or k >= 15:
                        break
                    ax[j][i].imshow(x[h][w][k], cmap='gray')
                    ax[j][i].set_title('Mask:' + str(w) + '-train:' + str(h), fontsize=7)
                    k += 1
            plt.savefig("output_images/train" + str(h) + "/output" + str(w) + ".png")
            plt.close(fig)
