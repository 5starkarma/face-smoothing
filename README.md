# Face Smoothing: Face Detection and Smoothing

OpenCV implementation of facial smoothing. Facial detection is done using an pretrained TensorFlow face detection model. Facial smoothing is accomplished using the following steps:

- Switch image to HSV colorspace
- Create mask of HSV image
- Apply a bilateral filter to the Region of Interest
- Apply filtered ROI back to original image

---

Input Image             |  Output Image w/ Face Smoothing
:-------------------------:|:-------------------------:
![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/images/hillary_clinton.jpg?raw=true "Input image")  |  ![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/output/output_with_bbox0.jpg?raw=true "Output image")
