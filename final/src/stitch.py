#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:36:21 2020

@author: suhelbeer
"""
import cv2
import os
import sys
import glob

import numpy as np





def matcher(des1,des2,kp1,kp2):
    #this function computes the matches between two images and returns the index list of good matches
    # the inputs are descriptors and keypoints of the two images to be matched
    R= len(des1)
    C= len(des2)
    s=(R,C)
    dist= np.zeros(s)
    for i in range(R):
        for j in range(C):
            dist[i][j]= np.linalg.norm(des1[i]-des2[j])
    
    #using SSD to compute the distances
    
    indexlist= []
    img1co=[]
    img2co=[]
    distemp= np.sort(dist,axis=1)
    for i in range(R):
        #filtering out good matches
        if distemp[i][0] < 0.75*distemp[i][1]:
            minin= np.argmin(dist[i]) 
            ind= (i,minin)
            indexlist.append(ind)
            img1co.append(kp1[i].pt)
            img2co.append(kp2[minin].pt)
            

    
    return indexlist


def RANSAC(matches,kpdes,kpsrc):
    #this function computes homography by using RANSAC to find the points which have minimum number of inliers
    maxinlier=0
    homography= None
    r=len(matches)
    for _ in range(10):
       #this runs for 10 iterations 
        np.random.shuffle(matches)
       
        for i in range(0,len(matches)-4,4):  
            pt1=[]
            pt2=[]
            for j in range(i,i+4):
                a= matches[j][0]
                b= matches[j][1]
               # x= (kpdes.pt)
               # y= (kpsrc.pt)
                pt1.append(kpdes[a].pt)
                pt2.append(kpsrc[b].pt)
            
            ptd1= np.float32(pt1)
            pts2= np.float32(pt2)
            h= cv2.getPerspectiveTransform(pts2,ptd1)
           #getting homography h for 4 random points and then calculating the inliers
            inlier=0
            for pts in range(r):
                x=matches[pts][0]
                y=matches[pts][1]
                ptsrc=kpsrc[y].pt
                ptdes=kpdes[x].pt
                ps=[ptsrc[0],ptsrc[1],1]
                trans= np.dot(h,ps)
                trans= np.matrix(1/trans[2]*trans)
                
              
                pd=np.matrix([ptdes[0],ptdes[1],1])
               
                diff = np.linalg.norm(np.subtract(pd,trans))
                if diff < 2:
                   
                    inlier= inlier+1
                
            if inlier>maxinlier:
                maxinlier=inlier
                if maxinlier>4:
                    homography=h
              #updating the results for maxinliers found and the corresponding homography matrix  
    return homography
            
                
                
            
def stich(imgdes,imgsrc,H):
    #this function takes two images as input along with their homography matrix and returns the stitched image
    imgsize= (imgsrc.shape[1]+imgdes.shape[1]+imgsrc.shape[1],imgdes.shape[0]+imgsrc.shape[0]+imgsrc.shape[0])
    pano1= cv2.warpPerspective(imgsrc,H,imgsize)
    pano1[imgsrc.shape[0]:imgsrc.shape[0]+imgdes.shape[0],imgsrc.shape[1]:imgsrc.shape[1]+imgdes.shape[1]]=imgdes
    return pano1
 


def panorama(pano,img,lenpano,a):
    #this function sorts and stitches multiple input images. It looks for the images having similar features and then stitches them together
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    
    r=len(img)
    maxmat=0
    # for finding the image with maximum mathces with the given image 
    for i in range (0,r): 
        kpp, desp = sift.detectAndCompute(pano, None)
        kpi, desi = sift.detectAndCompute(img[i], None)
        dis = len(matcher(desi,desp,kpi,kpp))
        if dis > maxmat:
            maxmat=dis
            imgnext=img[i]
            p=i
    #if maximum matches are greater than 10 then it will stitch those two images together    
    if maxmat>10:
        imgt= cv2.copyMakeBorder(imgnext,pano.shape[0],0,pano.shape[1],0,cv2.BORDER_CONSTANT)
        kpp, desp = sift.detectAndCompute(pano, None)
        kpi, desi = sift.detectAndCompute(imgt, None)
        dis = matcher(desi,desp,kpi,kpp)
        homography=RANSAC(dis,kpi,kpp)
        pano= stich(imgnext,pano,homography)
        lenpano+=1
        img.pop(p)
        if len(img)>0: 
            pano=panorama(pano,img,lenpano,a)
        
            
    #if maximum matches are less than 10 and length of panorama is 1 then it will pick up next image to find the panorama       
    elif maxmat<10 and lenpano==1:
        pano=img[a+1]
        img.pop(a+1)
        pano= panorama(pano,img,1,a+1)
    #if maximum amatches are less tha  10 and length of panorama is greater than 1 then it will return the panorama
    elif maxmat<10 and lenpano>1:
        return pano
    return pano
    
    
    

def main():
    #inputs the image directory
    path_to_directory= (os.path.join(os.getcwd(),sys.argv[1]))

    image_paths = glob.glob(path_to_directory + "/*.jpg")

    #reads all the images from the given directory and ignores the image named panorama
    imgt = [cv2.imread(image_path) for image_path in image_paths if "panorama" not in image_path]
   
    
    
    
    
    w = int(imgt[0].shape[1])
    h = int(imgt[0].shape[0]* 400/w)
    dsize= (400,h)
    img=[cv2.resize(imgc,dsize) for imgc in imgt]
    #resizing all images 
    
    
    img=img[::-1]
    #image 1 is taken as initial panorama and then the function is called 
    pano=img[0]
    img.pop(0)
    lenpano= 1
    #calling the panorama function to calculate the panorama and return the result 
    pano=panorama(pano,img,lenpano,0)

 
   
    #writes the final output to the directory where the images are stored
    
  
    cv2.imwrite(os.path.join(path_to_directory,"panorama.jpg"),pano)
   
    
if __name__ == "__main__":
    main()




