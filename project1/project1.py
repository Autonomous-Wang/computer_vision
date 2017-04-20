#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        # for instance, if channel is 3, the ignore color is (255, 255, 255)
        ignore_mask_color = (255,) * channel_count 
    else:
        ignore_mask_color = 255  
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    # if the image is gray, dstack it into 3 dimensional image
    if len(img.shape) == 2:  
        img = np.dstack((img, img, img))
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 >= 0 and x1 < img.shape[1] and \
                y1 >= 0 and y1 < img.shape[0] and \
                x2 >= 0 and x2 < img.shape[1] and \
                y2 >= 0 and y2 < img.shape[0]:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # img should be the result of canny detection and return the image with lines draw on it
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha = 0.8, beta = 1., lamda = 0.):
    # pipeline should return initial_img * α + img * β + λ
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)

def rgb_to_hsv(img):
    """ Converts an RGB image to a HSV image"""    
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def show_img(img):
    """ Displays an image, no matter if it is RGB or single channel"""
    if len(img.shape) > 2:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

def preprocess(img):
    """ Preprocess the input image """
    # Convert to HSV and get value (luminance) channel, in order to make more robust
    hsv = rgb_to_hsv(img)
    v = hsv[:,:,2]
    
    # Apply Gaussian Blur to reduce the noise in edge detection
    kernel_size = 5
    out = gaussian_blur(v, kernel_size)   
    return out

def apply_canny(img):
    """ Applies the Canny edge detector to the input image """
    # Apply Canny edge detector
    low_threshold = 50
    high_threshold = 150
    
    out_img = canny(img, low_threshold, high_threshold)
    return out_img

def select_region_of_interest(img):
    # select four verices and then fill a polygon into the image
    h = 20
    k = 1.75
    v1 = (0 + h, img.shape[0]) 
    v2 = (img.shape[1]/2 - 5, img.shape[0]/k) 
    v3 = (img.shape[1]/2 + 5, img.shape[0]/k) 
    v4 = (img.shape[1] - h, img.shape[0])
    return region_of_interest(img, np.array([[v1, v2, v3, v4]], dtype=np.int32))
    
def extract_edges(img):
    # Get edges using the Canny edge detector
    img_canny = apply_canny(img)    
    return select_region_of_interest(img_canny)
    
def detect_lines(img_canny_masked):
    """ Runs the Hough transform to detect lines in the input image"""
    # Apply HoughLines to extract lines
    rho_res         = 2                 # [pixels]
    theta_res       = np.pi/180.        # [radians]
    threshold       = 60                # [# votes]
    min_line_length = 120                # [pixels]
    max_line_gap    = 100                # [pixels]
    lines = cv2.HoughLinesP(img_canny_masked, rho_res, theta_res, threshold, np.array([]), 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def fitLine(line_points):
    # based on (x1,y1,x2,y2), compute the line equation y = mx + b
    x1 = line_points[0]
    y1 = line_points[1]
    x2 = line_points[2]
    y2 = line_points[3]
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return (m, b)

def extract_lanemarkings(img_shape, lines):
    # given a list of lines, calculate the average slope and intersection
    # For each line segment, we set the range for each line slope
    # the main idea is to gather all the line points and then average
    
    slope_min = 0.5
    slope_max = 2.0
  
    m1 = np.array([])
    b1 = np.array([])
    
    m2 = np.array([])
    b2 = np.array([])

    y_min = img_shape[0]
    
    # in each line points, the points are (x1, y1, x2, y2)
    for line_points in lines:
        # Fit to line equation (m, b)
        (m, b) = fitLine(line_points)

        # Filter line by slope
        if abs(m) > slope_min and abs(m) < slope_max:
            y_min = min(y_min, line_points[1])
            y_min = min(y_min, line_points[3])
        
            # Separate into left/right using the sign of the slope
            if (m > 0):
                m1 = np.append(m1, m)
                b1 = np.append(b1, b)
            
            else:
                m2 = np.append(m2, m)
                b2 = np.append(b2, b)
        
    # Average the two main lines and get the equation
    m1 = np.mean(m1)
    b1 = np.mean(b1)
    
    m2 = np.mean(m2)
    b2 = np.mean(b2)
    
    # Compute the crossing (x,y) point in the image
    x_cross = (b2 - b1) / (m1 - m2)
    y_cross = m1 * x_cross + b1
    
    # End point of the line: at most the crossing point
    y_end = min(y_cross, y_min)
    
    # Compute the (x) coordinate where the line crosses the 
    # bottom edge of the image
    y1 = img_shape[0] - 1
    x1 = (y1 - b1) / m1
    y2 = img_shape[0] - 1
    x2 = (img_shape[0] - b2) / m2    
    
    x_end1 = (y_end - b1) / m1
    x_end2 = (y_end - b2) / m2
    
    return np.array([[[x1, y1, x_end1, y_end]], [[x2, y2, x_end2, y_end]]]).astype(int)

def overlay_lanemarkings(img, lanemarkings):
    """ Draws the lines on top of the image img """
    # Create a black image with red lanemarkings
    img_lines = np.copy(img)*0
    draw_lines(img_lines, lanemarkings, color=[255, 0, 0], thickness=10)
    
    # Blend the original image with the previous one
    img_out = weighted_img(img_lines, img)
    return img_out

def pipeline(img_original):
    """
    Process the input image 'img' and outputs an annotated version of it,
    where the left and right lane markings are detected.
    """
    # Pre-process gaussian blur of the luminance color space, output the image
    img = preprocess(img_original)
    
    # Extract edges
    # canny detection plus mask, bitwise operation
    img_edges = extract_edges(img)
    
    # Detect lines, employ the hough transform
    lines = detect_lines(img_edges)   
    
    # Extract left and right lanemarkings from the lines, return the four end-points
    lanemarkings = extract_lanemarkings(img.shape, lines.squeeze())
    
    # Produce output
    img_out = overlay_lanemarkings(img_original, lanemarkings)    
    return img_out

import os
test_images = os.listdir("test_images/")

if not os.path.isdir("results"):
    os.mkdir("results")

for i in range(0, len(test_images)):
    # Read image
    img = mpimg.imread(os.path.join("test_images", test_images[i]))
    
    # Run the pipeline
    img_out = pipeline(img)
    plt.subplot(2,3,i + 1)
    show_img(img_out)
    plt.axis('off')

    # Save output
    mpimg.imsave(os.path.join("results", test_images[i]), img_out) 

# test on video stream
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = pipeline(image)
    return result

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


















