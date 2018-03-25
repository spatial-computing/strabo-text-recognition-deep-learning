from scipy import ndimage
from scipy.misc import toimage
import cv2

def rotateImage(image, angle):
    rotated = ndimage.rotate(image, angle)
    toimage(rotated).show()
    return rotated

rotateImage(cv2.imread('1920-3.png') ,45)

    