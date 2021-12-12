import cv2
import numpy as np
import math
import sys
import time
import Adafruit_BBIO.PWM as PWM
# import detectRed
#from detectRed import isTrafficStop
#from detectRed import isGreenLight

#Throttle
throttlePin = "P8_13"
go_forward = 7.955
dont_move = 7.5

#Steering
steeringPin = "P9_14"
left = 9
right = 6

#Max number of loops
max_ticks = 2000

#Booleans for handling stop light
passedStopLight = False
atStopLight = False



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
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_boundaries = redBoundary[0]
    percentage = redBoundary[1]
    lower = np.array(color_boundaries[0])
    upper = np.array(color_boundaries[1])
    mask = cv2.inRange(hsv_img, lower, upper)
    output = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    # print(np.count_nonzero(mask), mask.size)
    percentage_detected = np.count_nonzero(mask) * 100 / np.size(mask)
    print("percentage_detected " + str(percentage_detected) + " lower " + str(lower) + " upper " + str(upper))

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



def initialize_car():
    # give 7.5% duty at 50Hz to throttle
    print("starting function")
    PWM.start(throttlePin, dont_move, frequency=50)
    print("did the pwm")

    input()
    PWM.start(steeringPin, dont_move, frequency=50)


def stop():
    PWM.set_duty_cycle(throttlePin, dont_move)


def go():
    PWM.set_duty_cycle(throttlePin, go_forward)


def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV",hsv)
    lower_blue = np.array([90, 120, 0], dtype = "uint8")
    upper_blue = np.array([150, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    #cv2.imshow("mask",mask)

    # detect edges
    edges = cv2.Canny(mask, 50, 100)
    #cv2.imshow("edges",edges)

    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array([[
        (0, height),
        (0, height/2),
        (width , height/2),
        (width , height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, mask)
    #cv2.imshow("roi",cropped_edges)

    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1
    theta = np.pi / 180
    min_threshold = 10

    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments

def average_slope_intercept(frame, line_segments):
    lane_lines = []

    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity")
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape

    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down

    if slope == 0:
        slope = 0.1

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def get_steering_angle(frame, lane_lines):

    height,width,_ = frame.shape

    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)

    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)

    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle

initialize_car()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

time.sleep(1)

##fourcc = cv2.VideoWriter_fourcc(*'XVID')
##out = cv2.VideoWriter('Original15.avi',fourcc,10,(320,240))
##out2 = cv2.VideoWriter('Direction15.avi',fourcc,10,(320,240))

speed = 8
lastTime = 0
lastError = 0

kp = 0.07
kd = kp * 0.6

counter = 0
go()

while counter < max_ticks:
    # check for stop sign/traffic light every couple ticks

    ret,original_frame = video.read()
    
    frame = cv2.resize(original_frame, (160, 120))

    #if ((counter + 1) % 3) == 0:
    #    print("checking for stop light?")
    #    if not passedStopLight and not atStopLight:
    #        trafficStopBool, _ = isTrafficStop(frame)
    #        print(trafficStopBool)
    #        if trafficStopBool:
    #            print("detected red light, stopping")
    #            stop()
    #            atStopLight = True
    #            continue
    
    #if not passedStopLight and atStopLight:
    #    print("waiting at red light")
    #    trafficGoBool, _ = isGreenLight(frame)
    #    if trafficGoBool:
    #        passedStopLight = True
    #        atStopLight = False
    #        print("green light!")
    #        go()
    #    else:
    #        continue

    #cv2.imshow("original",frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame,line_segments)
    lane_lines_image = display_lines(frame,lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    #heading_image = display_heading_line(lane_lines_image,steering_angle)
    #cv2.imshow("heading line",heading_image)
    # floorRed = detectRed.isFloorStop(frame)[1]
    # trafficRed = detectRed.isTrafficStop(frame)[1]
    # cv2.imshow("FloorStop[left] trafficStop[right]", np.hstack([floorRed, trafficRed]))
    now = time.time()
    dt = now - lastTime

    deviation = steering_angle - 90
    # error = abs(deviation)

    ### PD Code, remove if breaking things
    error = -deviation
    base_turn = 7.5
    proportional = kp * error
    derivative = kd * (error - lastError) / dt

    turn_amt = base_turn + proportional + derivative

    if turn_amt > 7.2 and turn_amt < 7.8:
        # May not need this condition
        turn_amt = 7.5
    elif turn_amt > left:
        turn_amt = left
    elif turn_amt < right:
        turn_amt = right

    PWM.set_duty_cycle(steeringPin, turn_amt)
    print(turn_amt)


    ### END PD Code
    ### Old steering code
    # if deviation < 5 and deviation > -5:
    #     deviation = 0
    #     error = 0
    #     PWM.set_duty_cycle(steeringPin, 7.5)

    # elif deviation > 5:
    #     PWM.set_duty_cycle(steeringPin, 6)


    # elif deviation < -5:
    #     PWM.set_duty_cycle(steeringPin, 9)

    # derivative = kd * (error - lastError) / dt
    # proportional = kp * error
    # PD = int(speed + derivative + proportional)
    # spd = abs(PD)

    # if spd > 25:
    #     spd = 25
    ### END Old steering code

    # throttle.start(spd) - we keep the speed low and constant

    lastError = error
    lastTime = time.time()

##    out.write(frame)
##    out2.write(heading_image)

    key = cv2.waitKey(1)
    if key == 27:
        break

    counter += 1

video.release()
##out.release()
##out2.release()
cv2.destroyAllWindows()
PWM.set_duty_cycle(throttlePin, 7.5)
PWM.set_duty_cycle(steeringPin, 7.5)
PWM.stop(throttlePin)
PWM.stop(steeringPin)
PWM.cleanup()


test = 0
if test:
    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while (1):
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
