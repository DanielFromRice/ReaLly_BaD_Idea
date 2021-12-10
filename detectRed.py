import cv2
import numpy as np



def getImage():
    image = cv2.imread("red.png")
    return image

def getRedBoundaries():
    return getBoundaries("redboundaries.txt")

def isFloorStop(frame):
    """
    Detects whether or not the floor is red
    :param frame: Image
    :return: (True is the camera sees a stop light on the floor, false otherwise) and video output
    """
    boundaries = getRedBoundaries()
    return isMostlyColor(frame, boundaries)

def isMostlyColor(image, redBoundary):
    """
    Detects whether or not the majority of a color on the screen is a particular color
    :param image:
    :param redBoundary:
    :return:
    """
    lower = np.array(redBoundary[0])
    upper = np.array(redBoundary[1])
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    # print(np.count_nonzero(mask), mask.size)
    result = (np.count_nonzero(mask) > (mask.size - np.count_nonzero(mask)))
    return result, output

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def getBoundaries(filename):
    with open(filename, "r") as f:
        boundaries = f.readlines()
        lower = [int(val) for val in boundaries[0].split(",")]
        upper = [int(val) for val in boundaries[1].split(",")]
        boundaries = [lower, upper]
    return boundaries


while(1):

    ret, frame = video.read()
    if ret == False:
        continue
    frame = cv2.flip(frame, -1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # cv2.imshow("original", frame)
    floorStop, output = isFloorStop(frame)
    if floorStop:
        print(True)
    else:
        print(False)
    cv2.imshow("images", output)
    key = cv2.waitKey(1)
    if key == 27:
        break
