
# IVDC AUTONOMY

This is a code for a Problem statement code which involves designing a pipeline which processes a given image of a lane and produces a clear bitmap output where:

1.The area between the lanes is marked black.

2.The surrounding regions are marked white.

Additionally,it also isolates and extracts the background of the given image and subtracts it from the original image.



## Description(Approach)
### STEP 1: IMPORT:- Prompting the user to select an image of interest.  
```
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tiff)'; ...
                                  '*.*', 'All Files (*.*)'}, 'Select an Image');
```

### STEP 2: PREPROCESSING:    

#### Step A: Conversion of color image to grayscale image.    

```   
code:
grayImage = rgb2gray(img);  
```
We prefer grayscale image over a colour image because it has only one color channel(reduced data size) which increases the computation speed.Also it requires less memeory storage(occupies a third of the space required by RGB images). 

#### Step B: Contrast adjustment   
```
code:   
con = imadjust(grayImage);   
```   
Contrast adjustment is crucial for enhancing image quality and visibility particularly in grayscale images.High contrast images make it easier to identify features and boundaries.   

####  Step C:Noise Removal using an averaging filter.  
```   
F = fspecial("average", 5); 
eqFil = imfilter(con, F,"replicate");   
```    
here  "replicate" avoids the dark border lines.   
This step is important as it helps in reducesing the number of edges detected in region of interest which is the lane.This will compliment the fact that we should keep the lane black as the morphological closing operation expands 
the white regions and we dont want it to happen in the lane region.

### STEP 3: SEGMENTATION   
Its the process of dividing an image into multiple regions or segments such that each region is homogeneous in some sense.   
#### Step A: Edge Detection   
```   
code:   
edges = edge(eqFil);
```   
Here the "edge" function detects all the edges in our preprocessed image.
The edge detection method involves canny,sobel,prewitt,roberts etc..   
*getting error with these methods*    

### STEP 4: POST-PROCESSING       
Involves the application of morphological functions like dilation and erode coupled which includes closing and opening morphological functions.   
In "dilation" the brightest pixel in the window is taken.   
In "erosion" the darkest pixel in the window is taken.
"Opening" morphological function ,it removes all small white patterns preserving the larger ones.     
"Closing" morphological function ,it removes all small dark patterns preserving the larger ones. 

```   
code(opening):        
str=strel("disk",5);
open=imopen(edges,str);   
close=imclose(edges,str);
```   
Here str is a structuring element which in this case is a disk of radius 5.The Structuring element acts as a sliding window and examines each part of the image. And the shape of the str also has effect on the pattern of the bordera and edges.    



#### Additional step used here:  
We cropped out the upper half of the image which mainly includes the sky/background ,and then again we use morphological closing function to further isolate the lane and remove all the outliers.


## Concepts Explored:  
### Hough Transform:     
Hough Line Transform: The Hough transform is a feature extraction technique used in image analysis to find shapes within an object. Here, it takes each edge pixel in the image and maps to a collection of lines surrounding possible values within some or all parameters space. We use generally parameters such as slope(m) and y-intercept(b).

 Here is a simplification of the process:
 * Edge Detection: In the first step, we process input image by finding edges on it with some edge detection method like Canny Edge Detector.

* Parameter Space: It generates a parameter space where every point in the form of lines represents a potential candidate line passing through the original images.

* Voting: For each edge pixel its coordinates are examined and a list of possible lines is created. Every line represents a point in the parameter space. The "votes" are for the line of fpts.

* Peak Detection: Search the parameter space for peaks, i.e. lines (lines which got voted by most people) or! These peaks shows the line s that you have most probably in the real image.
The Hough transform is a powerful tool for line detection in various applications, including lane detection, object detection, and document analysis.

*wasnt able to implement this in the code*
## Results
![Lane Detection 1](https://github.com/UnceasingEagerness/IVDC_Autonomy/blob/main/Result%201.png)

![Lane Detection 1](https://github.com/UnceasingEagerness/IVDC_Autonomy/blob/main/Result%202.png)   
Above are the few glimpses of what the code is capable of doing!

