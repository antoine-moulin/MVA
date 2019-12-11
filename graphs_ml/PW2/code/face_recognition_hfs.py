import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import cv2
import os

from helper_online_ssl import *
from harmonic_function_solution import *


def offline_face_recognition(plot_results=True):
    """
    TO BE COMPLETED

    Function to test offline face recognition.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)

    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and equalization (cv.equalizeHist).
            """
            clahe = cv2.createCLAHE(tileGridSize=(8, 8))
            # gray_face = cv2.equalizeHist(gray_face)
            gray_face = clahe.apply(gray_face)

            gray_face = cv2.medianBlur(gray_face, 3)
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf

            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False

    if plot_the_dataset:
        plt.figure(1)
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:6]
        for m in mask:
            mlabels[m * 10 + i] = 0

    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 1000
    eps = 0
    k = 15
    laplacian_regularization = gamma
    laplacian_normalization = "rw"
    c_l = .9
    c_u = .1

    # hard or soft HFS
    # rlabels = hard_hfs(images, mlabels, laplacian_regularization, var, eps, k, laplacian_normalization)
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    # Plots #
    accuracy = np.equal(rlabels, labels).mean()
    if plot_results:
        plt.subplot(121)
        plt.imshow(labels.reshape((10, 10)))

        plt.subplot(122)
        plt.imshow(rlabels.reshape((10, 10)))
        plt.title("Acc: {}".format(accuracy))

        plt.show()

    return accuracy

    
def offline_face_recognition_augmented(plot_results=True):
    """
    TO BE COMPLETED.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    nbimgs = 50
    # Loading images
    images = np.zeros((10 * nbimgs, frame_size ** 2))
    labels = np.zeros(10 * nbimgs)

    for i in np.arange(10):
        imgdir = "data/extended_dataset/%d" % i
        imgfns = os.listdir(imgdir)
        for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
            im = imread("{}/{}".format(imgdir, imgfn))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            clahe = cv2.createCLAHE(tileGridSize=(8, 8))
            # gray_face = cv2.equalizeHist(gray_face)
            gray_face = clahe.apply(gray_face)

            gray_face = cv2.medianBlur(gray_face, 3)
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf

            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False
    if plot_the_dataset:

        plt.figure(1)
        for i in range(10 * nbimgs):
            plt.subplot(nbimgs, 10, i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    nb_to_reveal = 20
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(nbimgs)
        np.random.shuffle(mask)
        mask = mask[:nbimgs-nb_to_reveal]
        for m in mask:
            mlabels[m * 10 + i] = 0


    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 1000
    eps = 0
    k = 25
    laplacian_regularization = gamma
    laplacian_normalization = "rw"
    c_l = .95
    c_u = .05
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    # Plots #
    accuracy = np.equal(rlabels, labels).mean()
    if plot_results:
        plt.subplot(121)
        plt.imshow(labels.reshape((-1, 10)))

        plt.subplot(122)
        plt.imshow(rlabels.reshape((-1, 10)))
        plt.title("Acc: {}".format(accuracy))

        plt.show()

    return accuracy


if __name__ == '__main__':
    # # question 2.2 and 2.3 #
    # print('----- Question 2.2 and 2.3 -----')
    # nb_measures = 10
    # accuracies = np.zeros((nb_measures,))
    #
    # # offline_face_recognition()  # to have one plot
    # for k in range(nb_measures):
    #     accuracies[k] = offline_face_recognition(plot_results=False)
    #
    # print('On an average of {} measures, the accuracy obtained is: {} (std: {})\n'.format(nb_measures,
    #                                                                                     accuracies.mean(),
    #                                                                                     accuracies.std()))

    # question 2.4 #
    print('----- Question 2.4 -----')
    nb_measures = 10
    accuracies = np.zeros((nb_measures,))

    offline_face_recognition_augmented()
    # for k in range(nb_measures):
    #     accuracies[k] = offline_face_recognition_augmented(plot_results=False)
    #
    # print('On an average of {} measures, the accuracy obtained is: {} (std: {})'.format(nb_measures,
    #                                                                                     accuracies.mean(),
    #                                                                                     accuracies.std()))
