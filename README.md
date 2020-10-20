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

---
![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/output/combined.jpg?raw=true "Processing steps")

## Install
```
git clone https://github.com/5starkarma/face-smoothing.git
cd face-smoothing
```
## Run
```
python3 infer.py --input 'path/to/input_file.jpg' (Input file - default is hillary_clinton.jpg)
                 --output 'path/to/output_folder' (Output folder - default is data/output)
                 --save_steps 'path/to/file.jpg' (Concats images from each step of the process and saves them)
```

## TODO
- [ ] Fix combined output
- [ ] Test on multiple faces
- [ ] Video inference
- [ ] Unit tests
- [ ] Test better blurring techniques
- [ ] Run time tests on units
