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
#from sklearn import datasets
#img_path = glob.glob('C:/Users/15418/Downloads/Data (March 2022)/2022_tree_trunk_measurements/*/')
img_path=glob.glob('C:/Users/15418/Downloads/Trunk cross sectional area/Trunk cross sectional area/data_tree/*/')
#variable to switch the img and ptcloud at the same time
imgind=4
outliers=[563,566]
#11,260
#no mask for 2,7,18,19,22
#imglist=(0,1,3,4,5,6,8,9,10,11,12,13,14,15,16,17,20,21,23,24,25)
imglist=(0,3,4,6,7,8,9,10,11,13,14,15,16,17,18,21,22,23,25,29,30,31,32,33,36,38,39,40,41,42,43,44,45,46,47,49,51,53,56,58,59,60,61,63,64,68,69,70,71,72,74,75,77,80,81,83,84,85,86,87,88,89,90,91,92,93,95,96,97,98,99)
#weird: 2,5,19,20, 21?,26,28,34

#73 has a point cloud that is entirely NaN
#heightlist=(245,250,220,135,260,285,225,280,255,260,205,235,300,210,235,240,225,200,205,260,255)
heightlist=(395,190,135,210,125,25,165,210,115,170,125,105,10,155,170,330,220,360,240,265,240,170,300,370,350,335,125,290,50,390,240,300,405,240,250,275,340,295,305,175,185,145,130,120,215,140,200,280,130,205,140,200,190,190,180,145,230,185,125,130,125,160,315,280,300,300,285,300,230,230,280,290,235,235,220,280,320,270,260,345,290,310,240,300,335,355,290,295,360,345,345,285,330,290,330,305,305,275,290,295)
#treenums=[560,561,562,562,563,563,564,565,565,566,566,566,567,567,568,568,570,570,571,572,572]
#truewidths=(.0643,.0636,.0684,.0684,.0688,.0688,.0706,.0724,.0724,.0913,.0913,.0913,.0666,.0666,.0778,.0778,.0759,.0759,.0688,.0675,.0675)
image1 = cv2.imread((glob.glob(img_path[imglist[imgind]] + '/*.png'))[0])
#first image is tree 560
pt_cloud=(glob.glob(img_path[imgind] + '/*.ply'))[0]
pc = read_ply.read_ply(pt_cloud)
pc = pc.reshape(480,640, 6)
pc = pc[:, :, 0:3]
ROOT_DIR = os.path.abspath("../")
classes = ['Graft union']
#test_model, inference_config = m_rcnn.load_inference_model(1, "C:/Users/15418/PycharmProjects/thesis/mask rcnn2/mask_rcnn_3data.h5")
#r = test_model.detect([image1])[0]
img = image1.copy()
colors = [255, 0, 0]
#object_count = len(r["class_ids"])
#mask1 = r["masks"][:, :, 0]
width=640
#width=len(mask1[0])
class Slice(object):
    def __init__(self,height,width,mask,pc,image):
        self.height = height
        self.width=width
        self.mask=mask
        self.pc=pc
        self.image=image
        self.width1=[]
        self.width2=[]
    def GetWidth(self):
        colors = [255, 0, 0]
        contours = get_mask_contours(self.mask)
        for cnt in contours:
            cv2.polylines(self.image, [cnt], True, colors[0], 2)
            img = draw_mask(self.image, [cnt], colors[0])
        count = 0
        start=0
        zs = []
        xs = []
        for i in range(self.width):
            if self.mask[self.height,i]==True and math.isnan(self.pc[self.height,i][0])==False:
                if count==0:
                    start=i
                x=self.pc[self.height,i][0]
                z=self.pc[self.height,i][2]
                zs.append(z)
                xs.append(x)
                count=count+1
        plt.axis("equal")
        plt.plot(xs,zs)
        #print(zs)
        depth=np.average(zs)
        #print(depth)
        self.width1=np.max(xs)-np.min(xs)
        #print("width1: ", width1)
        picwidth=np.tan(np.deg2rad(34.5))*depth*2
        self.width2=picwidth*count/width
        #print("width2: ",width2)
        #plt.show()
        cv2.line(self.image, (start, self.height), (start + count, self.height), (0, 0, 250), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        loc = (start+count+20, self.height)
        fontScale = .5
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2
        print("len(pc)")
        cv2.putText(self.image, 'W1: ' + str(round(self.width1,3)) + " W2: " + str(round(self.width2,3)), loc,  font,fontScale,fontColor,  thickness, lineType)
        cv2.imshow("img",self.image)
        cv2.waitKey(0)
def plotMask(mask, pc, image):
    zs=[]
    for i in range(len(mask)):
        for q in range(len(mask[0])):
            if mask[i,q]==True:
                if math.isnan(pc[i,q][0])==True:
                    pass
                else:
                    z=pc[i,q][2]
                    zs.append(z)
    zmax=max(zs)
    zmin=min(zs)
    for i in range(len(mask)):
        for q in range(len(mask[0])):
            if mask[i,q]==True:
                if math.isnan(pc[i,q][0])==True:
                    color=(255,255,255)
                else:
                    z=pc[i,q][2]
                    color=(round(255*(z-zmin)/(zmax-zmin)),0,round(255*(z-zmin)/(zmax-zmin)))
                image[i,q]=color
                #print(color)
    cv2.imshow("mask",image)
    cv2.waitKey(0)
def heatmap():
    for i in range(len(imglist)):
        s=imglist[i]
        print(s)
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        # first image is tree 560
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        pc = read_ply.read_ply(pt_cloud)
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        r = test_model.detect([image])[0]
        mask = r["masks"][:, :, 0]
        plotMask(mask,pc,image)

def debugest():
    treenums=gettreenums(imglist)
    s =74
    i=50
    image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
    # first image is tree 560
    pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
    pc = read_ply.read_ply(pt_cloud)
    pc = pc.reshape(480, 640, 6)
    pc = pc[:, :, 0:3]
    # I know that there's a better way to do this, glob wasn't working
    if treenums[i] == treenums[i - 1] and treenums[i] != treenums[i - 2]:
        maskname = 'mask' + str(treenums[i]) + 'a' + '.npy'
    elif treenums[i] == treenums[i - 1] and treenums[i] == treenums[i - 2]:
        maskname = 'mask' + str(treenums[i]) + 'b' + '.npy'
    else:
        maskname = 'mask' + str(treenums[i]) + '.npy'
    mask = np.load(maskname)
    plotMask(mask, pc, image)
    rangew1 = []
    rangew2 = []
    rangew3 = []
    # average of 10 slices across 20 pixels around the height
    for g in range(-10, 10, 2):
        a = PCAslice(heightlist[s] + g, width, mask, pc, image)
        #for i in range(0,639):
         #   print(mask[heightlist[s] + g,i])
        a.GetWidth()
def truthVsEst():
    w1=[]
    w2=[]
    w3=[]
    treenums=gettreenums(imglist)
    truewidths=getwidths()
    for i in range(len(imglist)):
        s=imglist[i]
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        # first image is tree 560
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        pc = read_ply.read_ply(pt_cloud)
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        #I know that there's a better way to do this, glob wasn't working
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            maskname='mask' + str(treenums[i])+'a' +'.npy'
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            maskname='mask' + str(treenums[i]) + 'b' +'.npy'
        else:
            maskname='mask'+str(treenums[i]) +'.npy'
        mask=np.load(maskname)
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
        #a = PCAslice(heightlist[i], width, mask, pc, image)
        #a.GetWidth()
        #cv2.imshow(str(s),a.image)
        #cv2.waitKey(0)
        w1av = np.average(rangew1)
        w2av = np.average(rangew2)
        w3av = np.average(rangew3)
        #print("w1 error: ", a.width1-truewidths[i])
        #print("w2 error: ", a.width2-truewidths[i])
        #print("w3 error: ", a.width3-truewidths[i])
        if w1av>.3:
            print(s)
        w1.append(w1av)
        w2.append(w2av)
        w3.append(w3av)
    linex=[.04,.1]
    liney=[.04,.1]
    #print(w3)
    np.save("w1", w1)
    np.save("w2", w2)
    np.save("w3", w3)
    w1error=np.sum(np.abs(np.subtract(w1,truewidths)))
    w2error=np.sum(np.abs(np.subtract(w2,truewidths)))
    w3error = np.sum(np.abs(np.subtract(w3, truewidths)))
    print("w1 total error:", w1error)
    print("w2 total error:", w2error)
    print("w3 total error:", w3error)
    plt.plot(truewidths,w1,'g^',truewidths,w2,'r*',truewidths,w3,'bo',linex,liney)
    plt.legend(["width1", "width2","width3"])
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    plt.show()
    plt.plot(truewidths,w2,'r*',truewidths,w3,'bo',linex,liney)
    plt.legend(["width2","width3"])
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    plt.show()

def getCenter(mask):
    #empty arrays for xs and ys
    xs=[]
    ys=[]
    for i in range(0,len(mask),15):
        for q in range(len(mask[0])):
            if mask[i,q]==True:
                xs.append(q)
                ys.append(i)
                break
    slopes=[]
    for i in range(len(xs)-1):
        x1=xs[i]
        x2=xs[i+1]
        y1=ys[i]
        y2=ys[i+1]
        if x1!=x2:
            slope=((y2-y1)/(x2-x1))
        else:
            slope=10000
        slopes.append(slope)
    return xs,ys,slopes

def drawSlopes(mask,image):
    [xs,ys,slopes]=getCenter(mask)
    print("xs: ",xs)
    print("ys: ", ys)
    seglen=15
    for i in range(len(slopes)):
        if slopes[i]!=10000:
            angle=math.atan(slopes[i])
            xc=int(math.cos(angle)*seglen)
            yc=int(math.sin(angle)*seglen)
        else:
            xc=0
            yc=seglen
        #print(slopes[i])
        cv2.line(image, (xs[i], ys[i]), (xs[i] + xc, ys[i]+yc), (0, 0, 250), 1)
    for i in range(len(slopes)):
        if slopes[i]!=10000:
            angle = math.atan(slopes[i])
            if angle<0:
                perpangle = angle + np.pi / 2
            else:
                perpangle = angle - np.pi / 2
            xc=int(math.cos(perpangle)*seglen)
            yc=int(math.sin(perpangle)*seglen)
        else:
            xc=seglen
            yc=0
        cv2.line(image, (xs[i], ys[i]), (xs[i] + xc, ys[i]+yc), (0, 0, 250), 1)
    for i in range(len(slopes)):
        testx=xs[i]
        print(slopes[i])
        testy = ys[i]
        truex=testx
        truey=testy
        while mask[testy,testx]==True:
            if slopes[i]!=10000:
                angle = math.atan(slopes[i])
                perpangle = angle + np.pi / 2
                xc=math.cos(perpangle)
                yc=math.sin(perpangle)
            else:
                xc=1
                yc=0
            truex=truex+xc
            truey=truey+yc
            testx=int(truex)
            testy=int(truey)
        #step back to the last True reading
        truex = truex - xc
        truey = truey - yc
        cv2.line(image, (xs[i], ys[i]), (int(truex), int(truey)), (0, 255, 0), 1)
    cv2.imshow("lines", image)
    cv2.waitKey(0)
#drawSlopes(mask1,image1)
class angledSlice(object):
    def __init__(self,height,width,mask,pc,image):
        self.height = height
        self.width=width
        self.mask=mask
        self.pc=pc
        self.image=image
        self.width1=1
        self.width2=[]
    def GetWidth(self):
        colors = [255, 255, 0]
        contours = get_mask_contours(self.mask)
        for cnt in contours:
            cv2.polylines(self.image, [cnt], True, colors[0], 2)
            img = draw_mask(self.image, [cnt], colors[0])
        count = 0
        start=0
        [xs, ys, slopes] = getCenter(self.mask)
        #start list of depth values
        zs=[]
        print(ys)
        for i in range(len(ys)):
            if self.height<ys[i]:
                slope=slopes[i]
                break
        #print(slope)
        #should change to find starting x,y and then run loop
        for i in range(self.width):
            if self.mask[self.height, i] == True:
                startx=i
                break
        testx = startx
        testy = self.height
        truex = testx
        truey = testy
        realxs=[]
        realys=[]
        while self.mask[testy, testx] == True:
            if slope != 10000:
                angle = math.atan(slope)
                perpangle = angle + np.pi / 2
                xc = math.cos(perpangle)
                yc = math.sin(perpangle)
            else:
                xc = 1
                yc = 0
            truex = truex + xc
            truey = truey + yc
            testx = int(truex)
            testy = int(truey)
            z=self.pc[testy,testx][2]
            x = self.pc[testy, testx][0]
            y = self.pc[testy, testx][1]
            if math.isnan(x) == False:
                realxs.append(x)
            if math.isnan(y)==False:
                realys.append(y)
            if math.isnan(z)==False:
                zs.append(z)
        truex = truex - xc
        truey = truey - yc
        x1 = startx
        y1 = self.height
        x2 = int(truex)
        y2 = int(truey)
        linelength = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        cv2.line(self.image, (x1, y1), (x2, y2), (40, 120, 100), 1)
        realx1=min(realxs)
        realx2=max(realxs)
        realy1=realys[0]
        realy2=realys[len(realys)-1]
        self.width1 = ((realx2-realx1)**2+(realy2-realy1)**2)**.5
        cv2.line(self.image, (x1, y1), (x2, y2), (255, 120, 0), 1)
        depth=np.average(zs)
        picwidth=np.tan(np.deg2rad(34.5))*depth*2
        self.width2=picwidth*linelength/width
        font = cv2.FONT_HERSHEY_SIMPLEX
        loc = (start+count+20, self.height)
        fontScale = .5
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2
        cv2.putText(self.image, 'W1: ' + str(round(self.width1,3)) + " W2: " + str(round(self.width2,3)), loc,  font,fontScale,fontColor,  thickness, lineType)
        cv2.imshow("img",self.image)
        cv2.waitKey(0)
#a=angledSlice(200,width,mask1,pc,image1)
#drawSlopes(mask1,image1)
def pctesting(pc,image):
    close_pts_x = []
    close_pts_y = []
    print("len(pc): ", len(pc))
    for i in range(len(pc)):
        for q in range((len(pc[i]))):
            x = q
            #the point clouds are upside down
            y = i
            z = pc[i,q][2]
            if .3 < z < 2:
                close_pts_x.append(x)
                close_pts_y.append(y)
    plt.axis("equal")
    plt.scatter(close_pts_x, close_pts_y)
    plt.show()
    print(len(image))
    cv2.imshow("img",image)
    cv2.waitKey(0)

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)

    angle = math.atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * math.cos(angle)
    q[1] = p[1] - scale * hypotenuse * math.sin(angle)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

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
class PCAslice(object):
    def __init__(self,height,width,mask,pc,image):
        self.height = height
        self.width=width
        self.mask=mask
        self.pc=pc
        self.image=image
        self.width1=1
        self.width2=[]
        self.width3=[]
        self.imheight=[]
        self.depth=[]
        self.slicewidth=[]
    #xs=[]
    #ys=[]
    #pts=[]
    #for i in range(len(mask)):
       # for q in range(len(mask[0])):
       #     if mask[i,q]==True:
      #          xs.append(q)
     #           ys.append(i)
    #for i in range(len(xs)):
    #    pts.append([xs[i],ys[i]])
    #pca = PCA(n_components=2)
    #pca.fit(pts)
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    def GetWidth(self):
        contours = get_mask_contours(self.mask)
        for i, c in enumerate(contours):
            a=getOrientation(c, self.image)
        #cv2.imshow("img", self.image)
        #cv2.waitKey(0)
        ang2=a-np.pi/2
        startx=0
        for i in range(self.width):
            if self.mask[self.height, i] == True:
                startx=i
                break
        #for debugging
        if startx ==0:
            for cnt in contours:
                cv2.polylines(self.image, [cnt], True, colors[0], 2)
                img = draw_mask(self.image, [cnt], colors[0])
            print(self.height)
            cv2.imshow("img", self.image)
            cv2.waitKey(0)
        testx = startx
        testy = self.height
        #print("start: ", testx, testy)
        truex = testx
        truey = testy
        realxs=[]
        realys=[]
        zs=[]
        xc = math.cos(ang2)
        yc = math.sin(ang2)
        while self.mask[testy, testx] == True:
            truex = truex + xc
            truey = truey + yc
            testx = round(truex)
            testy = round(truey)
            #print(truex,truey,"new: ",testx,testy)
            z = self.pc[testy, testx][2]
            x = self.pc[testy, testx][0]
            y = self.pc[testy, testx][1]
            #print(testy,testx,x)
            #print(self.mask[testy, testx])
            if math.isnan(x) == False:
                realxs.append(x)
            if math.isnan(y) == False:
                realys.append(y)
            if math.isnan(z) == False:
                zs.append(z)
        #plt.axis("equal")
        #plt.plot(realxs, zs)
        #plt.show()
        #step back to true reading
        truex = truex - xc
        truey = truey - yc
        x1 = startx
        y1 = self.height
        x2 = int(truex)
        y2 = int(truey)
        linelength = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5
        realx1 = min(realxs)
        realx2 = max(realxs)
        realy1 = realys[0]
        realy2 = realys[len(realys) - 1]
        self.width1 = ((realx2 - realx1) ** 2 + (realy2 - realy1) ** 2) ** .5
        cv2.line(self.image, (x1, y1), (x2, y2), (20, 200, 40), 1)
        depth = np.average(zs)
        picwidth = np.tan(np.deg2rad(34.5)) * depth * 2
        self.slicewidth=linelength
        self.width2 = picwidth * linelength / self.width
        font = cv2.FONT_HERSHEY_SIMPLEX
        loc = (x1 + int(linelength) + 20, self.height)
        fontScale = .5
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2
        #new array to store all z values
        zs2=[]
        for i in range(len(self.pc)):
            for q in range(len(self.pc[0])):
                z=self.pc[i,q][2]
                if math.isnan(z) == False:
                    zs2.append(z)
        totaldepth=np.mean(zs2)
        #print(totaldepth)
        totalwidth=np.tan(np.deg2rad(34.5)) * totaldepth * 2
        rat=picwidth/totalwidth
        imheight=depth*np.tan(np.deg2rad(21))*2
        self.imheight=imheight
        self.depth=depth
        distperpix=imheight/480
        self.width3=linelength*distperpix
        #cv2.putText(self.image, 'W1: ' + str(round(self.width1, 3)) + " W2: " + str(round(self.width2, 3)) + " W3: " + str(round(self.width3, 3)), loc, font,
         #           fontScale, fontColor, thickness, lineType)
        #projection approach (underestimates)
        #self.width3=self.width2*rat
        #cv2.imshow("img", self.image)
        #cv2.waitKey(0)

#truthVsEst()
#a=PCAslice(260,width,mask1,pc,image1)
#a.GetWidth()
#cv2.imshow("img",a.image)
#v2.waitKey(0)
#point cloud gets called y,x

def maskSaving():
    treenums = gettreenums(imglist)
    for i in range(len(imglist)):
        s=imglist[i]
        print(s)
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        r = test_model.detect([image])[0]
        mask = r["masks"][:, :, 0]
        colors = [255, 0, 0]
        contours = get_mask_contours(mask)
        for cnt in contours:
            cv2.polylines(image, [cnt], True, colors[0], 2)
            img = draw_mask(image, [cnt], colors[0])
        #cv2.imshow("img"+str(s),image)
        #cv2.waitKey(0)
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            np.save('mask' + str(treenums[i])+'a', mask)
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            np.save('mask' + str(treenums[i]) + 'b', mask)
        else:
            np.save('mask'+str(treenums[i]), mask)

def treeAngles():
    angles=[]
    ss=[]
    for i in range(len(imglist)):
        s=imglist[i]
        ss.append(s)
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        # first image is tree 560
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        pc = read_ply.read_ply(pt_cloud)
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        #I know that there's a better way to do this, glob wasn't working
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            maskname='mask' + str(treenums[i])+'a' +'.npy'
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            maskname='mask' + str(treenums[i]) + 'b' +'.npy'
        else:
            maskname='mask'+str(treenums[i]) +'.npy'
        mask=np.load(maskname)
        hlist=[]
        for q in range(len(pc)):
            for t in range(len(pc[0])):
                if mask[q,t]==True:
                    hlist.append(q)
                    break
        hmax=max(hlist)
        hmin=min(hlist)
        dep=[]
        realh=[]
        for h in range(0,10,2):
            for w in range(len(pc[0])):
                if mask[hmin+h,w]==True:
                    d=pc[hmin+h,w][2]
                    y=pc[hmin+h,w][1]
                    if np.isnan(d)==False:
                        realh.append(y)
                        dep.append(d)
        topheight=np.average(realh)
        topdepth=np.average(dep)
        realh=[]
        dep=[]
        for h in range(0,10,2):
            for w in range(len(pc[0])):
                if mask[hmax-h,w]==True:
                    d=pc[hmax-h,w][2]
                    y=pc[hmax-h,w][1]
                    if np.isnan(d)==False:
                        realh.append(y)
                        dep.append(d)
        bottomdepth=np.average(dep)
        bottomheight = np.average(realh)
        angle=np.rad2deg(np.arctan((topheight-bottomheight)/(topdepth-bottomdepth)))
        angles.append(angle)
    plt.plot(ss,angles,'r*')
    plt.show()

def imHeightvsTreedepth():
    slices=[]
    imheights=[]
    ss=[]
    ratio=[]
    for i in range(len(imglist)):
        s=imglist[i]
        ss.append(s)
        image = cv2.imread((glob.glob(img_path[s] + '/*.png'))[0])
        pt_cloud = (glob.glob(img_path[s] + '/*.ply'))[0]
        pc = read_ply.read_ply(pt_cloud)
        pc = pc.reshape(480, 640, 6)
        pc = pc[:, :, 0:3]
        #I know that there's a better way to do this, glob wasn't working
        if treenums[i]==treenums[i-1] and treenums[i]!=treenums[i-2]:
            maskname='mask' + str(treenums[i])+'a' +'.npy'
        elif treenums[i]==treenums[i-1] and treenums[i]==treenums[i-2]:
            maskname='mask' + str(treenums[i]) + 'b' +'.npy'
        else:
            maskname='mask'+str(treenums[i]) +'.npy'
        mask=np.load(maskname)
        a = PCAslice(heightlist[i], width, mask, pc, image)
        a.GetWidth()
        imheights.append(a.imheight)
        slices.append(a.slicewidth)
        rat=a.imheight/a.slicewidth
        ratio.append(rat)
    plt.plot(ratio,truewidths,'ro')
    for i in range(len(ss)):
        plt.annotate(str(ss[i]),(ratio[i],truewidths[i]))
    plt.show()
def getwidths():
    trunkwidths1=[]
    with open('C:/Users/15418/PycharmProjects/thesis/groundtruth.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            num=row['Average']
            trunkwidths1.append(float(num))
    trunkwidths = []
    for i in range(len(imglist)):
        ind=imglist[i]
        trunkwidths.append(trunkwidths1[ind])
    trunkwidths=np.array(trunkwidths)/1000
    return trunkwidths

def gettreenums(imglist):
    treenums=[]
    for i in range(len(imglist)):
        ind=imglist[i]
        if ind==1 or ind==0:
            num=202
        else:
            num=ind+201
        treenums.append(num)
    return treenums
#treeAngles()
#maskSaving()
#imHeightvsTreedepth()
#a=getwidths()
#debugest()
#truewidths = getwidths()
#print(truewidths)
#truthVsEst()
#check: 8,10,11,23,58
def r2():
    w3=np.load('w3.npy')
    w2=np.load('w2.npy')
    truewidths=getwidths()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(truewidths,w3)
    print('R^2: ', r_value**2)

treenums=gettreenums(imglist)
s=imglist[imgind]
maskname = 'mask' + str(treenums[imgind]) + '.npy'
mask1 = np.load(maskname)
colors = [255, 0, 0]
object_count = 1
for i in range(object_count):
    # 1. Mask
    mask = mask1
    contours = get_mask_contours(mask)
    for cnt in contours:
        cv2.polylines(image1, [cnt], True, colors[i], 2)
        img = draw_mask(image1, [cnt], colors[i])
a = PCAslice(heightlist[s], width, mask1, pc, image1)
a.GetWidth()
cv2.imshow("img", a.image)
cv2.waitKey(0)
#r2()
