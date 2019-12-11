import cv2 as cv
import os
import numpy as np


face_haar_cascade = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_haar_cascade = cv.CascadeClassifier("data/haarcascade_eye.xml")


def create_user_profile(user_name, faces_path="data/"):
    """
    Uses the camera to collect data.
    :param user_name: name that identifies the person/face
    :param faces_path: where to store the images
    """
    ## Find the "faces" directory
    # assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    image_count = 0
    if not os.path.exists(profile_path):
        os.makedirs(profile_path)
        print("New profile created at path", profile_path)
    else:
        image_count = len(os.listdir(profile_path))
        print("Profile found with", image_count, "images.")
        ## Launch video capture
    cam = cv.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        grey_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        working_image = cv.bilateralFilter(grey_image, 9, 75, 75)
        working_image = cv.equalizeHist(working_image)
        working_image = cv.GaussianBlur(working_image, (5, 5), 0)
        box = face_haar_cascade.detectMultiScale(working_image)
        if len(box) > 0:
            box_surface = box[:, 2] * box[:, 3]
            index = box_surface.argmax()
            b0 = box[index]
            cv.rectangle(img, tuple([b0[0] - 4, b0[1] - 4]), tuple([b0[0] + b0[2] + 4, b0[1] + b0[3] + 4]), (0, 255, 0),
                         2)
        cv.putText(img, "[s]ave file, [e]xit", (5, 25), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        cv.imshow("cam", img)
        key = cv.waitKey(1)
        if key in [27, 101]: break  # esc or e to quit
        if key == ord('s'):
            ## Save face
            if len(box) > 0:
                x, y = b0[0], b0[1]
                x_range, y_range = b0[2], b0[3]
                image_count = image_count + 1
                image_name = os.path.join(profile_path, "img_" + str(image_count) + ".bmp")
                img_to_save = img[y:(y + y_range), x:(x + x_range)]
                cv.imwrite(image_name, img_to_save)
                print("Image", image_count, "saved at", image_name)
    cv.destroyAllWindows()
    return


def load_profile(user_name, faces_path="data/"):
    """
    Loads the data associated to user_name.

    Returns an array of shape (number_of_images, n_pixels)
    """
    assert ("faces" in os.listdir(faces_path)), "Error : 'faces' folder not found"
    ## Check if profile exists. If not, create it.
    faces_path = os.path.join(faces_path, "faces")
    profile_path = os.path.join(faces_path, user_name)
    if not os.path.exists(profile_path):
        raise Exception("Profile not found")
    image_count = len(os.listdir(profile_path))
    print("Profile found with", image_count, "images.")
    images = [os.path.join(profile_path, x) for x in os.listdir(profile_path)]
    rep = np.zeros((len(images), 96 * 96))
    for i, im_path in enumerate(images):
        im = cv.imread(im_path, 0)
        cv.waitKey(1)
        rep[i, :] = preprocess_face(im)
    return rep


def preprocess_face(grey_face):
    """
    Transforms a n x n image into a feature vector
    :param grey_face: ( n x n ) image in grayscale
    :return gray_face_vector:  ( 1 x EXTR_FRAME_SIZE^2) row vector with the preprocessed face
    """
    # Face preprocessing
    EXTR_FRAME_SIZE = 96
    """
     Apply preprocessing to balance the image (color/lightning), such    
      as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and 
      equalization (cv.equalizeHist).                                     
    """
    grey_face = cv.bilateralFilter(grey_face, 9, 75, 75)
    grey_face = cv.equalizeHist(grey_face)
    grey_face = cv.GaussianBlur(grey_face, (5, 5), 0)

    # resize the face
    grey_face = cv.resize(grey_face, (EXTR_FRAME_SIZE, EXTR_FRAME_SIZE))
    grey_face = grey_face.reshape(EXTR_FRAME_SIZE * EXTR_FRAME_SIZE)
    # scale the data to [0,1]
    grey_face = grey_face / 256
    return grey_face
