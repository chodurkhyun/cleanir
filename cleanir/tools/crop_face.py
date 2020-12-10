import face_recognition
import cv2
import numpy as np

def crop_face_from_image(image, dsize):
    """Detect and crop the largest face in the image

    Arguments:
        image {np.array} -- input image
        dsize {tuple} -- cropped face's size (w, h)

    Return:
        cropped face image
    """

    faces = face_recognition.face_locations(image, model="cnn")

    if len(faces) > 0:
        t, r, b, l = max(faces,
                         key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
        return cv2.resize(image[t:b, l:r], dsize)
    else:
        return cv2.resize(image, dsize)


def crop_face_from_file(filepath, dsize):
    """Detect and crop the largest face in the image
    
    Arguments: filepath {str} -- image file path
               dsize {tuple} -- cropped face's size (w, h)

    Return:
        cropped face image
    """

    image = face_recognition.load_image_file(filepath)
    return crop_face_from_image(image, dsize)


def crop_face_from_image_fit(image, dsize):
    """Cut a face in the image using facial landmarks

    Arguments:
        image {np.array} -- input image
        dsize {tuple} -- cropped face's size (w, h)
    """

    face_landmarks_list = face_recognition.face_landmarks(image)
    if len(face_landmarks_list) < 1: return None
    landmarks = [c for val in face_landmarks_list[0].values() for c in val if c[0] >= 0 if c[1] >= 0]
    l, t, w, h = cv2.boundingRect(np.array(landmarks))

    return cv2.resize(image[t:t + h, l:l + w], dsize)


def crop_face_from_file_fit(filepath, dsize):
    """Cut a face in the image

    Arguments:
        filepath {str} -- image file path
        dsize {tuple} -- cropped face's size (w, h)

    Return:
        cropped face image
    """

    image = face_recognition.load_image_file(filepath)
    return crop_face_from_image_fit(image, dsize)
