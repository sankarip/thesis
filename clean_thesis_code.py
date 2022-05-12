#This code was written by Pico Sankari for his honors thesis
#If there are an issues with it, feel free to email me at picosankari1@gmail.com
#import all relevant libraries
#make sure to download all the files off the github and put them in the same folder
#weights can be downloaded here: https://drive.google.com/drive/folders/11RuLe0ALHT_de5gANapfLnKCLh29aBCl?usp=sharing
#weights must be in the folder the rest of the code is run from
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import m_rcnn
from visualize import draw_mask, get_mask_contours
import os
import read_ply
import math
import csv
import scipy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
#this is the path the ground truth data. It will need to be updated to whatever your local path is
img_path=glob.glob('C:/Users/15418/Downloads/Trunk cross sectional area/Trunk cross sectional area/data_tree/*/')
#imglist is the list of the indices of images that produce decent masks with the current mask-RCNN. 8 is an outlier that could be removed
#this list will need to be updated if the mask-RCNN is retrained
imglist=(0,3,4,6,7,8,9,10,11,13,14,15,16,17,18,21,22,23,25,29,30,31,32,33,36,38,39,40,41,42,43,44,45,46,47,49,51,53,56,58,59,60,61,63,64,68,69,70,71,72,74,75,77,80,81,83,84,85,86,87,88,89,90,91,92,93,95,96,97,98,99)
#heightlist is the height (in pixels) of all of the slices for all 100 ground truth trees
heightlist=(395,190,135,210,125,25,165,210,115,170,125,105,10,155,170,330,220,360,240,265,240,170,300,370,350,335,125,290,50,390,240,300,405,240,250,275,340,295,305,175,185,145,130,120,215,140,200,280,130,205,140,200,190,190,180,145,230,185,125,130,125,160,315,280,300,300,285,300,230,230,280,290,235,235,220,280,320,270,260,345,290,310,240,300,335,355,290,295,360,345,345,285,330,290,330,305,305,275,290,295)
#classes for the graft union detector
classes = ['Graft union']
#makes all the masks blue
colors = [255, 0, 0]
#width of images
width=640

#function that helps create heatmaps of your point clouds. Useful for looking at errors or noise in the point cloud
def plotMask(mask, pc, image):
    #empty array for Z (depth) values
    zs=[]
    #runs across every point in the cloud
    for i in range(len(mask)):
        for q in range(len(mask[0])):
            #if the point is part of the mask
            if mask[i,q]==True:
                #if the point has a nan value
                if math.isnan(pc[i,q][0])==True:
                    #do nothing
                    pass
                # if the point isnt nan
                else:
                    #get z value. Calling values from the point cloud can be confusing, the indexing is [Y][X]
                    z=pc[i,q][2]
                    #add the z value to the array
                    zs.append(z)
    #find the furthest and closest points
    zmax=max(zs)
    zmin=min(zs)
    #go across the image
    for i in range(len(mask)):
        for q in range(len(mask[0])):
            #if the pixel is part of the mask
            if mask[i,q]==True:
                #if the pixel is nan
                if math.isnan(pc[i,q][0])==True:
                    #change the color to white
                    color=(255,255,255)
                else:
                    #change the pixel to a color between magenta and black depending on how far away it is
                    #color range could be redone, but this works
                    z=pc[i,q][2]
                    color=(round(255*(z-zmin)/(zmax-zmin)),0,round(255*(z-zmin)/(zmax-zmin)))
                #set the pixel to the appropriate color
                image[i,q]=color
    #show the image. Could be turned off and the image could be saved if you wnat to do a lot of these
    cv2.imshow("mask",image)
    cv2.waitKey(0)

#function that makes a heatmap of every image on the image list
def heatmap():
    #load the mask RCNN. You will have to update the path to your local path
    test_model, inference_config = m_rcnn.load_inference_model(1,"C:/Users/15418/PycharmProjects/thesis/mask rcnn2/mask_rcnn_3data.h5")
    for i in range(len(imglist)):
        #get the indice of the next image
        s=imglist[i]
        #read the image
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        #get the point cloud
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        #read the point cloud
        pc = read_ply.read_ply(pt_cloud)
        #reshape the point cloud
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        #You can update the following lines to just load masks so that the code runs quicker. See later functions for examples
        #run the network
        r = test_model.detect([image])[0]
        #get the mask
        mask = r["masks"][:, :, 0]
        #use the previous function to plot the masks
        #you could probably consolidate the two functions
        plotMask(mask,pc,image)

#function that gets the tree numbers based on the image list
#only made to work with the current ground truth
def gettreenums(imglist):
    #empty array
    treenums=[]
    #go down the list
    for i in range(len(imglist)):
        ind=imglist[i]
        #the first two are tree 202
        if ind==1 or ind==0:
            num=202
        #after that they just count up
        else:
            num=ind+201
        #add to array
        treenums.append(num)
    #return the array
    return treenums

#function that gets the widths from the .csv
def getwidths():
    trunkwidths1=[]
    #open the file
    with open('C:/Users/15418/PycharmProjects/thesis/groundtruth.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #get the value under the header 'Average'
            num=row['Average']
            #turn it to a float
            trunkwidths1.append(float(num))
    #new array to only take the widths you want
    trunkwidths = []
    #get the widths that correspond with the trees on the image list
    for i in range(len(imglist)):
        ind=imglist[i]
        trunkwidths.append(trunkwidths1[ind])
    #convert millimeters to meters
    trunkwidths=np.array(trunkwidths)/1000
    return trunkwidths

#function that makes and saves masks for future use
def maskSaving():
    #get the tree numbers
    treenums = gettreenums(imglist)
    #load the network. Must be changed to your own weights path
    test_model, inference_config = m_rcnn.load_inference_model(1,"C:/Users/15418/PycharmProjects/thesis/mask rcnn2/mask_rcnn_3data.h5")
    #go down the imglist
    for i in range(len(imglist)):
        #get the indice of the tree
        s=imglist[i]
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        r = test_model.detect([image])[0]
        mask = r["masks"][:, :, 0]
        #make it blue
        colors = [255, 0, 0]
        contours = get_mask_contours(mask)
        #draw the mask, good for debugging
        for cnt in contours:
            cv2.polylines(image, [cnt], True, colors[0], 2)
            img = draw_mask(image, [cnt], colors[0])
        #comment these lines once you're sure the network and images are loading correctly
        #just here for debugging. Image window has the indice of the tree in the name
        cv2.imshow("img"+str(s),image)
        cv2.waitKey(0)
        #save multiple masks from the same tree number (only works up to 3 different images for 1 tree) (could definetly be done in a more elegant way)
        #check if the tree has the same number as the 2 masks before it
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            np.save('mask' + str(treenums[i])+'a', mask)
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            np.save('mask' + str(treenums[i]) + 'b', mask)
        else:
            np.save('mask'+str(treenums[i]), mask)

#function that uses PCA to get the mask orientation. I did not write this code
def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0]
        data_pts[i, 1] = pts[i, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
    cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
    cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle

#function to visualize the results of PCA. I did not write this code
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    angle = math.atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * math.cos(angle)
    q[1] = p[1] - scale * hypotenuse * math.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

#function that takes slices angled by the PCA
class PCAslice(object):
    #initialize
    def __init__(self,height,width,mask,pc,image):
        #height at which the slice is taken
        self.height = height
        #width of the image
        self.width=width
        #mask for the image
        self.mask=mask
        #point cloud for the image
        self.pc=pc
        #the image
        self.image=image
        self.width1=1
        self.width2=[]
        self.width3=[]
        self.imheight=[]
        self.depth=[]
        self.slicewidth=[]
    def GetWidth(self):
        #pull up the mask
        contours = get_mask_contours(self.mask)
        for i, c in enumerate(contours):
            #get the orientation of the mask
            a=getOrientation(c, self.image)
        #get the angle perpendicular to the result (across the trunk)
        ang2=a-np.pi/2
        #find what x pixel the mask starts on
        startx=0
        for i in range(self.width):
            #if the mask is present
            if self.mask[self.height, i] == True:
                #this is the starting x pixel at the given height
                startx=i
                break
        #the pixels to start the slice at
        #testx and testy are used to check if the mask is still true (is the pixel part of the mask)
        #testx and testy have to be rounded to whole numbers
        testx = startx
        testy = self.height
        #truex and truey are used to keep track of the real slice value (don't have to be whole numbers)
        truex = testx
        truey = testy
        #arrays to store the point cloud values in
        realxs=[]
        realys=[]
        zs=[]
        #the increments that x and y change at across the slice
        xc = math.cos(ang2)
        yc = math.sin(ang2)
        #while the nearest pixel is part of the mask
        while self.mask[testy, testx] == True:
            #increment the real slice values
            truex = truex + xc
            truey = truey + yc
            #round these values to a whole number so that you can use them to check the mask
            testx = round(truex)
            testy = round(truey)
            #store the point cloud values of the nearest point
            z = self.pc[testy, testx][2]
            x = self.pc[testy, testx][0]
            y = self.pc[testy, testx][1]
            #only store them if they are numbers and not NAN
            if math.isnan(x) == False:
                realxs.append(x)
            if math.isnan(y) == False:
                realys.append(y)
            if math.isnan(z) == False:
                zs.append(z)
        #step back to the most recent true reading after exiting the loop
        truex = truex - xc
        truey = truey - yc
        #get the start and end points of the slice in pixels
        x1 = startx
        y1 = self.height
        x2 = int(truex)
        y2 = int(truey)
        #calculate the linelength in pixels
        linelength = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        #get the minimum and maximum x values from the point cloud
        realx1 = min(realxs)
        realx2 = max(realxs)
        #get the corresponding y values
        realy1 = realys[0]
        realy2 = realys[len(realys) - 1]
        #calculate the length of the slice based on point cloud values and assign to width 1
        self.width1 = ((realx2 - realx1) ** 2 + (realy2 - realy1) ** 2) ** .5
        #draw the line on the image to help visualize results
        cv2.line(self.image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        #get average depth of the slice
        depth = np.average(zs)
        #triangulate horiztonally
        picwidth = np.tan(np.deg2rad(34.5)) * depth * 2
        self.slicewidth=linelength
        #calvulate width 2
        self.width2 = picwidth * linelength / self.width
        #the following code writes the results next to the slice
        # the text overlaps if you take slices close together
        font = cv2.FONT_HERSHEY_SIMPLEX
        loc = (x1 + int(linelength) + 20, self.height)
        fontScale = .5
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2
        cv2.putText(self.image,
                    'W1: ' + str(round(self.width1, 3)) + " W2: " + str(round(self.width2, 3)) + " W3: " + str(
                        round(self.width3, 3)), loc, font,
                    fontScale, fontColor, thickness, lineType)
        imheight=depth*np.tan(np.deg2rad(21))*2
        #store the image height and depth for future reference
        self.imheight=imheight
        self.depth=depth
        #calculate the distance per pixel ratio
        distperpix=imheight/480
        #calculate the width 3 value
        self.width3=linelength*distperpix


#function that plots the estimated widths vs groundtruth
def truthVsEst():
    #empty arrays for each width
    w1=[]
    w2=[]
    w3=[]
    #get the tree numbers
    treenums=gettreenums(imglist)
    #get the widths from the CSV
    truewidths=getwidths()
    #go down the image list
    for i in range(len(imglist)):
        #get the indice from the image list
        s=imglist[i]
        #get the image
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        #get the point cloud
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        pc = read_ply.read_ply(pt_cloud)
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        #load masks
        #I know that there's a better way to do this, glob wasn't working
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            maskname='mask' + str(treenums[i])+'a' +'.npy'
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            maskname='mask' + str(treenums[i]) + 'b' +'.npy'
        else:
            maskname='mask'+str(treenums[i]) +'.npy'
        mask=np.load(maskname)
        #empty arrays that are used to get the average of the slices near the height
        rangew1=[]
        rangew2 = []
        rangew3 = []
        #average of 10 slices across 20 pixels around the height
        for g in range(-10,10,2):
            a = PCAslice(heightlist[s]+g, width, mask, pc, image)
            a.GetWidth()
            rangew1.append(a.width1)
            rangew2.append(a.width2)
            rangew3.append(a.width3)
        #average the results
        w1av = np.average(rangew1)
        w2av = np.average(rangew2)
        w3av = np.average(rangew3)
        #add the average to the results
        w1.append(w1av)
        w2.append(w2av)
        w3.append(w3av)
    #points that draw a line that represents where the ideal results would lie (slope of 1, intercept of 0)
    linex=[.04,.1]
    liney=[.04,.1]
    #save these values for quick reference in the future
    np.save("w1", w1)
    np.save("w2", w2)
    np.save("w3", w3)
    #calculate the total amount of error for each width
    w1error=np.sum(np.abs(np.subtract(w1,truewidths)))
    w2error=np.sum(np.abs(np.subtract(w2,truewidths)))
    w3error = np.sum(np.abs(np.subtract(w3, truewidths)))
    #print the results
    print("w1 total error:", w1error)
    print("w2 total error:", w2error)
    print("w3 total error:", w3error)
    #plot all 3 widths
    plt.plot(truewidths,w1,'g^',truewidths,w2,'r*',truewidths,w3,'bo',linex,liney)
    plt.legend(["width1", "width2","width3"])
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    plt.show()
    #plot just widths 2 and 3
    plt.plot(truewidths,w2,'r*',truewidths,w3,'bo',linex,liney)
    plt.legend(["width2","width3"])
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    plt.show()