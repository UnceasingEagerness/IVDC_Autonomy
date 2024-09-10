% prompting the user to select the image of choice
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp;*.tiff', 'Image Files (*.jpg, *.png, *.bmp, *.tiff)'; ...
                                  '*.*', 'All Files (*.*)'}, 'Select an Image');

% Check if the user selected a file or canceled
if isequal(filename,0)
    disp('User canceled the file selection.');
else
    %  Construct the file path and read the image
    originalImage = fullfile(pathname, filename);
    img = imread(originalImage);
end

% Read and display the image
a=zeros(50);
figure;
%originalImage = imread('C:/Users/Tanay/OneDrive/Documents/lane image.jpg');  
%originalImage = imread('C:/Users/Tanay/Downloads/IMG_20240906_142615.jpg');
grayImage = rgb2gray(img);
subplot(2,2,1)
imshow(img)
title("The original image")

% adjust contrast of the grayscale image using imadjust
con = imadjust(grayImage);

%Averaging filter to remove noise
F = fspecial("average", 5);  % Smaller filter size to avoid over-smoothing
eqFil = imfilter(con, F,"replicate");

% Edge detection using Canny
edges = edge(eqFil);
subplot(2,2,2)
imshow(edges)
title("Detected edges")

% Morphological closing with a line element 
s = strel('line',90, 0);  
%s = strel('disk',5);
edgeM = imclose(edges, s);

% Morphological closing with a disk element
s1 = strel('line', 10,0);  % Reduced size of disk element
edg = imclose(edgeM, s1);
%edg = edg(300:end,1:end);
el=strel("diamond",5);
edg1=imclose(edg,el);

%Cropping the upper half of the image
[rows, cols, ~] = size(edg1);
edg1(1:floor(rows/2), :, :) = 255;
s = strel('disk',18);  
%s = strel('disk',5);
final_lane = imclose(edg1, s);
subplot(2,2,3)
imshow(final_lane)
title("The Final Lane Bitmap")

maskedImage = img;  % Create a copy of the original image
maskedImage1 = img;
maskedImage(repmat(final_lane, [1 1 3])) = 255;
maskedImage1(repmat(final_lane, [1 1 3])) = 0;
%Extracted Background from the original image
backGround= img-maskedImage1;
subplot(2,2,4)
imshow(maskedImage)
title("Extracted Lane from Original Image")
















