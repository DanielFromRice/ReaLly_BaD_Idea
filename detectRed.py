import cv2
import numpy as np

def getImage():
    image = cv2.imread("red.png")
    return image

def getFloorRedBoundaries():
    return getBoundaries("redboundaries.txt")

def isFloorStop(frame):
    """
    Detects whether or not the floor is red
    :param frame: Image
    :return: (True is the camera sees a stop light on the floor, false otherwise) and video output
    """
    boundaries = getFloorRedBoundaries()
    return isMostlyColor(frame, boundaries)


def getTrafficStopBoundaries():
    return getBoundaries("trafficRedBoundaries.txt")


def isTrafficStop(frame):
    """
    Detects whether or not we can see a stop sign
    :param frame: 
    :return: 
    """
    
    boundaries = getTrafficStopBoundaries()
    return isMostlyColor(frame, boundaries)


def getTrafficGoBoundaries():
    return getBoundaries("trafficGreenboundaries.txt")


def isGreenLight(frame):
    """
    Detects whether or not we can see a green traffic light
    :param frame: 
    :return: 
    """
    boundaries = getTrafficGoBoundaries()
    return isMostlyColor(frame, boundaries)

def isMostlyColor(image, redBoundary):
    """
    Detects whether or not the majority of a color on the screen is a particular color
    :param image:
    :param redBoundary:
    :return:
    """
    color_boundaries = redBoundary[0]
    percentage = redBoundary[1]
    lower = np.array(color_boundaries[0])
    upper = np.array(color_boundaries[1])
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    # print(np.count_nonzero(mask), mask.size)
    percentage_detected = np.count_nonzero(mask) * 100 / np.size(mask)

    result = percentage[0] < percentage_detected <= percentage[1]
    if result:
        print(percentage_detected)
    return result, output


def getBoundaries(filename):
    """
    Reads the boundaries from the file filename
    Format:
        [0] lower: [H, S, V, lower percentage for classification of success]
        [1] upper: [H, S, V, upper percentage for classification of success]
    :param filename:
    :return:
    """
    default_lower_percent = 50
    default_upper_percent = 100
    with open(filename, "r") as f:
        boundaries = f.readlines()
        lower_data = [int(val) for val in boundaries[0].split(",")]
        upper_data = [int(val) for val in boundaries[1].split(",")]

        if len(lower_data) >= 4:
            lower_percent = lower_data[3]
        else:
            lower_percent = default_lower_percent

        if len(upper_data) >= 4:
            upper_percent = upper_data[3]
        else:
            upper_percent = default_upper_percent

        lower = lower_data[:3]
        upper = upper_data[:3]
        boundaries = [lower, upper]
        percentages = [lower_percent, upper_percent]
    return boundaries, percentages

test = 1
if test:
    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while(1):
        ret, frame = video.read()
        if ret == False:
            print("Erroring out")
            continue
        # frame = cv2.flip(frame, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # cv2.imshow("original", frame)
        floorStop, floor_output = isFloorStop(frame)
        trafficStop, traffic_output = isTrafficStop(frame)
        if trafficStop:
            print("Traffic stop detected!")
        trafficGo, traffic_green_output = isGreenLight(frame)
        if trafficGo:
            print("Traffic Go detected!")
        #
        # if floorStop:
        #     print(True)
        # else:
        #     print(False)
        cv2.imshow("images", np.hstack([traffic_output, traffic_green_output]))
        key = cv2.waitKey(1)
        if key == 27:
            break
