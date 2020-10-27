![alt text](https://travis-ci.org/5starkarma/face-smoothing.svg?branch=main "Build")

# Face Smoothing: Detection and Beautification

Input Image             |  Output Image w/ Facial Smoothing
:-------------------------:|:-------------------------:
![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/images/hillary_clinton.jpg?raw=true "Input image")  |  ![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/output/output_0.jpg?raw=true "Output image")
---
OpenCV implementation of facial smoothing. Facial detection is done using an pretrained TensorFlow face detection model. Facial smoothing is accomplished using the following steps:

- Change image from BGR to HSV colorspace
- Create mask of HSV image
- Apply a bilateral filter to the Region of Interest
- Apply filtered ROI back to original image

---

## Install
```
git clone https://github.com/5starkarma/face-smoothing.git
cd face-smoothing
```
## Run
```
python3 infer.py --input 'path/to/input_file.jpg' (Input file - image, video, or folder with images and/or videos - default is hillary_clinton.jpg)
                         'can/handle/videos.mp4'
                         'as/well/as/directories'
                 --output 'path/to/output_folder' (Output folder - default is data/output)
                 --save_steps 'path/to/file.jpg' (Concats images from each step of the process and saves them)
                 --show-detections (Saves bounding box detections to output)
```
#### Example: --save-steps flag
![alt text](https://github.com/5starkarma/face-smoothing/blob/main/data/output/combined_0.jpg?raw=true "Processing steps")

## TODO
- [X] Finish documentation and cleanup functions
- [X] Reduce input image size for detections
- [X] Fix combined output
- [X] Test on multiple faces
- [X] Apply blurring on multiple faces
- [X] Video inference
- [X] Save bounding box to output
- [ ] Apply different blurring techniques/advanced algo using facial landmarks to blur only skin regions
- [ ] Unit tests
- [ ] Run time tests on units
