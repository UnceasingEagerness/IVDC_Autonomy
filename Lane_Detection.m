% Read and display the image
a=zeros(50);
figure;
originalImage = imread('C:/Users/Tanay/OneDrive/Documents/lane image.jpg');  
%originalImage = imread('C:/Users/Tanay/Downloads/IMG_20240906_142615.jpg');
grayImage = rgb2gray(originalImage);

%  3x3 grid of subplots 
subplot(3,3,1)
imshow(grayImage);
title('Original Grayscale Image');

% adjust contrast of the grayscale image using imadjust
con = imadjust(grayImage);
subplot(3,3,2)
imshow(con)
title("Contrast Adjusted (imadjust)");

%Averaging filter to remove noise
subplot(3,3,3)
F = fspecial("average", 5);  % Smaller filter size to avoid over-smoothing
eqFil = imfilter(con, F);
imshow(eqFil);
title(' Smoothed Image');

% Edge detection using Canny
edges = edge(eqFil);
subplot(3,3,4)
imshow(edges);
title('Edges Detected (Smoothed Image)');

% Morphological closing with a line element 
s = strel('line',90, 0);  
%s = strel('disk',5);
edgeM = imclose(edges, s);
subplot(3,3,5)
imshow(edgeM);
title("Closed Image with Line Element");

% Morphological closing with a disk element
s1 = strel('line', 10,0);  % Reduced size of disk element
edg = imclose(edgeM, s1);
edg = edg(300:end,1:end);
el=strel("disk",5);
edg1=imclose(edg,el);
subplot(3,3,6)
imshow(edg1);
title("Closed Image with Disk Element");
%axis on
%axis image
%set(gca,'Color','w');
%pos=get(gca,'Position');
%xLimits=[0,1];
%yLimits=[0,1];
%set(gca,'XLim',xLimits,"YLim",yLimits);




% Binarize the equalized image
binaryImage = imbinarize(eqFil,"adaptive","ForegroundPolarity","dark","Sensitivity",0.85);
subplot(3,3,7)
imshow(binaryImage);
title('Binarized Image');

% Edge detection on the binarized image
edgesBinary = edge(binaryImage);
subplot(3,3,8)
imshow(edgesBinary);
title('Edges Detected on Binarized Image');

% Morphological operations: opening the binary image to remove noise
se = strel('disk', 5);
cleanedImage = imclose(binaryImage, se);
subplot(3,3,9)
imshow(cleanedImage);
title(' Binary Image');



