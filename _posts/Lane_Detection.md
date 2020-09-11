Autonomous driving is one of the most exciting and disruptive innovations of our time. Lane Detection is one of the preliminary steps involved during the training of an autonomous driving car. In this blog post, we will perform lane detection on [this video](https://drive.google.com/file/d/1V7OrAwjxtHR8LkfxXDVRu5lx2e5fxljo/view?usp=sharing).


```
# import the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Conversion to grayscale
There video frames are in RGB format. We'll convert this to grayscale because processing a single channel image is faster than processing a three-channel colored image.

### Image smoothening
Noise can create false edges. We will perform image smoothening using the `Gaussian` filter.

### Canny Edge Detector
The Canny Edge Detector computes gradient in all directions of the blurred image and traces the edges with large changes in intensity.  
The `Canny` function calculates derivative in both x and y directions, and according to that, we can see changes in intensity value. Larger derivatives equal high intensity, while smaller derivatives equal low intensity.


```
def canny_edge_detector(image):
    # convert the image color to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # reduce noise from the image
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
```

### Region of interest
We need to take into account onnly the region covered by the road lane. A mask is created, which is of the same dimension as our road image.  
A bitwise `AND` operation is performed between each pixel of our Canny image and the mask. It ultimately masks the Canny image and shows the region of interest traced by the polygonal contour of the mask.  
Let's mask our canny image after finding the region of interest.


```
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)

    # fill poly-function deals with multiple polygon
    cv2.fillPoly(mask, polygons, 255)

    # bitwise operation between canny image and mask image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
```

We are going to find the coordinates of our road lane.


```
def create_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])
```

We'll differentiate left and right road lanes with the help of positive and negative slopes respectively, and append them into the lists. If the sploe is negative, then the road lane belongs to the left-hand side of the vehicle, and if the slope is positive, then the road lane belongs to the right-hand side of the vehicle.


```
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # it will fit the polynomial and the intercept and slope
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = create_coordinates(image, left_fit_average)
    right_line = create_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
```

Fit the coordinates into our actual image and then return the image (i.e. road) with the detected line.


```
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
```

### Capturing and decoding the video file
We will capture the video using `VideoCapture` object. After capturing is initialized, every video frame is decoded and converted into a sequence of images.

### Hough Line Transform
This is a transform used to detect straight lines.

The video file is read and decoded into frames. Using the Hough Line Transform method, the straight line which is going through the image is detected.


```
# path of dataset directory
cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny_edge_detector(frame)
    cropped_image = region_of_interest(canny_image)

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                            np.array([]), minLineLength=40,
                            maxLineGap=5)

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("results", combo_image)

    # when the below two will be true and will press the 'q' on 
    # our keyboard, we will break out from the loop

    # wait 0 will wait for infinitely between each frames.
    # 1ms will wait for the specified time only between each frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close the video file
cap.release()

# destroy all the windows that is currently on
cv2.destroyAllWindows()
```

The output video can be viewed here.  

I have uploaded the input and output images of the program below.
