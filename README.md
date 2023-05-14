# Iris-Recongition-using-SIFT
*Project of the Computer Vision Course Offered in Spring 2022 @ Zewail City*

The project aims at introducing a model for iris detection and recognition using SIFT and BFMatcher. 
The process involves iris segmentation and detection using Canny Edge Detector and Hough Transform. The recognition process uses SIFT to extract local descriptors and then matches between iris images. 
The system was tested on well-known datasets: CASIA Iris 1 and UBIRIS Version 1.


## Model:

* The model takes two iris images as inputs and applies iris detection, segmentation, and SIFT feature extraction. 
* These outputs are compared and matched using BFMatch to determine if they belong to the same person. 
* The model can be generalized for multiple users and store feature vectors annotated by a unique ID. 
* When a new image is presented, the user must enter their ID, and the image is compared to the previously stored image associated with that ID.

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/8c2d4731-ea0c-4d5b-9199-5f1bce2e729d)

## Iris Detection:

The first stage of the model involves taking an input image obtained from iris scanning and identifying the boundaries of the iris. 
The iris is located between the pupil and the sclera, and this stage aims to extract only the iris which contains unique information.

Here are the steps of Iris Detection:
* Find the Pupil Circle using Hough Transform.
* Apply Median Filter to remove salt and pepper noise at the pupil.
* Apply Threshold Filter to binarize the colors in the image.
* Set different thresholds to enable Hough Transform to detect the pupil's circles.
* Use Hough Transform to obtain the Center X-coordinate, Center Y-coordinate, and Radius of the pupil's circles.
* Average all obtained values from all iterations to get the mean value.

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/16cfe331-6299-4e2b-9af3-b15b4c07cca4)


## Iris Segmetation:
In this stage, the model takes the locations and circles of both iris and pupils to extract the iris out of the image.

Here are the steps of Iris Segmetation:
* Model uses a Mask of 1’s at the location of iris.
* The mask is obtained using the circles of both pupils and iris.
* Bit-wise AND is used to segment the iris from the image.
* Histogram Equalization is used for color enhancement in order to get the utmost details from the iris.

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/33e59e0b-4083-48d1-8097-660d926e8eaf)

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/4d160c25-7bd1-4fff-8a38-1c3cadc27c4a)


## SIFT Features:
SIFT (Scale-Invariant Feature Transform) is a feature extraction technique that detects key points in an image and associates them with a signature to identify their appearance robustly to noise. It works by extracting frames of interest from the image and then filtering out variations. The process begins by extracting key points frame, followed by distance calculation for further filtration.

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/1b0805b6-2342-43bf-a965-5ad918ee19a6)

## BFMatch:
The model uses Braute-force matching with SIFT Descriptors.

Here are the steps of BFMatch:
* Braute-force with KNN is used to get k best matches, with k=2 in this case.
* Ratio Test is applied between output distances out of bf.knnMatches(), with a commonly used ratio of 0.8.
* Cosine Difference is applied for similarity between vertical angles of query and trained pictures.
* Median filtering is applied using the median technique.
* The standard deviation is tuned, with σ=100 as the standard deviation for vertical angle and σ=0.15 as the standard deviation for distances.

![image](https://github.com/ibrahimhamada/Iris-Recongition-using-SIFT/assets/58476343/aee6c42b-86c6-4209-9c55-d6f3ba683f4c)

