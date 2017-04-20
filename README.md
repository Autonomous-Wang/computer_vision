#Introduction:

the three folders contains the code for lane detection and traffic sign classification based on tensorflow

Project#1:
easy version of detection lane line, based on canny detection and hough transform, plus creating a image mask for region of interests. When we get the results from hough transform, we calculate the average value to make the results more robust.

Project#2:
advanced lane detection:
1. calibrate the camera based on the chessboard and then undistort the image.

2. construct image mask by thresholding different color space RGB and HLS. Besides, the sobel mask is also constructed to take the gradient into consideration and also transform the image into bird eye view (perspective transform).

3. collect the line pixel based on histogram analysis and sliding window to collect the lane pixels. and then we can polynomial fit the lane and calculate the curvature and the same time. Finally employ the pipeline to a video stream.

Project#3:
1. the main method is tensorflow, we feed the training set into the model and then get the trained model.

2. the training sets should be normalized before using, due to great numbers of training sets, we randomly split them into validation and test respectively and also divide the training sets into different batches.

3. the deep neuro network used is a Convolutional neuro network, due to it advantages in dealing with images.
