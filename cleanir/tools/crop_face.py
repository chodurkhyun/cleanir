import face_recognition
import cv2


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

    Arguments:
        filepath {str} -- image file path
        dsize {tuple} -- cropped face's size (w, h)

    Return:
        cropped face image
    """

    image = face_recognition.load_image_file(filepath)
    return crop_face_from_image(image, dsize)
