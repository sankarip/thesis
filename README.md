# Thesis
# Disclaimer: This code has not been optimized and could be restructured  
# The main code used for this thesis is in the file 'clean_thesis_code.py'
This is a cleaned up and commented version of the code used for this project. It's mostly broken into functions that complete various steps of the analysis.
To use this code you will need to download the maskcrnn utilities that are in this repo. You will also need the weights for the neural networks.
These can be found on google drive (link at the top of code) or box (https://oregonstate.box.com/s/yrkr0jgkl8ej6gkl65fut4229q7vmhts). 
Additonally, the file paths in lines 21, 79, 124, and 146 will need to be updated to your local paths for the code to work.
For more information on the project, consult the thesis.pdf document on box. 

# Recurring variables: imgpath, imglist, and heightlist
The way that the code is currently structured, imgpath is a path that leads to a folder that contains many other folders. Each folder within the main folder contains an image and a point cloud.
This is slighty convoluted, but is a functional way to deal with this data structure. 
imglist is a list of the images/point clouds on the path that you want to be analyzed. This is currently a lsit of all the images in the groundtruth data that generated complete masks.
The numbers on this list are the indices of the point clouds to be analyzed.
heightlist is a variable that contains the height of the tape on all 100 groundtruth images (even the ones not currently on imglist). 
These heights were manually assigned by opening the images in microsoft paint and placing the cursor appropriately.
These lists are currently just defined at the top of the code. This is a suboptimal way to structure the code and could be reworked.

# For all of the functions that take in masks, pointclouds, and images, the three need to be corresponding (image from point cloud, mask from image) or the code will not work correctly.

# Generating masks: maskingSaving() function 
No inputs, but requires a list of images and pointclouds (imglist and imgpath variables in the code) to be defined
It takes a long time to run the mask-RCNN on each image anytime you want to access its mask. This function uses imgpath and imglist to generate masks for each image on imglist.
These masks are saved as numpy variables that can easily be loaded for future use.

# The most important part of the code is the PCAslice class. 
This class takes is initialized with the height at which a slice should be taken, the width of the image, the appropriate trunk mask, the corresponding point cloud, and the image of the tree.
It returns 3 different width values. To get these widths, initialize the class and then use the GetWidth(self) function to add the widths as attributes of the class.

There are also various utilities in the code to help with finding widths of multiple images, etc.
# The function plotMask(mask, pc, image) takes in a mask, a point cloud, and an image
It returns a heatmap of the the mask. This can be helpful for debugging. The code breaks if there are any pixels in the mask that are very far away. 
The breaking means that the few far pixels will be magenta, while all other pixels are black without much a gradient. The NaN pixels will still be white.

# The function heatmap() 
No inputs, but requires a list of images and pointclouds (imglist and imgpath variables in the code) to be defined
This is a suboptimal way to structure the code and could be changed
This function returns heatmaps of all of the the images on the image list. Could be updated to save these images instead of displaying them if the user prefered.

# The function gettreenums(imglist) takes in the image list and returns the number for each tree on it
This function quickly places the correct tree number in a list that corresponds to the iamges listed on imglist. 

# The function getwidths() 
No inputs, but requires the path within the function to be changed to a path to a csv of the trunk widths and requires imglist and imgpath to be defined
An excel file with the trunk widths in it can be downloaded from box. This function quickly pulls the groundtruth data into python.

# getOrientation and drawAxis are used to perform and visualize the PCA analysis
I did not write these functions. An example of how to use them can be seen in the PCAslice class.

# The truthVsEst() function plots the groundtruth values against the estimates.
To use it, the path to the ground truth data must be properly updated and the maskSaving() function needs to have been used. Additionally, the imglist and imgpath variables must be correct.
The function will return 2 plots: one that contains all three widths and one that only conatins widths 2 and 3. The function will also print out the total error fro each width method.

# Retraining networks
I used this notebook to train the YOLO network: https://colab.research.google.com/drive/1zqRb08ljHvIIMR4fgAXeNy1kUtjDU85B?usp=sharing
I used this notebook to train the mask RCNN: https://colab.research.google.com/github/pysource7/utilities/blob/master/Train_Mask_RCNN_(DEMO).ipynb
All of the currently labeled data can be found on Box. For data labeling, I like to use https://www.makesense.ai/. To add to the current data sets, start by uploading them to makesense.ai (click 'Actions' then 'Import Images' and 'Import Annotations'). Next, upload the new images you would like to add to the data set and label them. It is important to load the old annotations before making new ones, you won't be able to bring in the old annotations after you start labeling without gettign rid of all the new annotations.
