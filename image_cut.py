# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 08:30:01 2021

@author: Asus
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('Insaat-20211105T065849Z-001/Insaat/13.jpg')
print(img.shape)
dh,dw,_ = img.shape

import glob
import os

image_list = []


def load_images_from_folder(folder):
    images = []
    
    for i in range(1,14):
        img = cv2.imread(folder+str(i)+".jpg")
        if img is not None:
            images.append(img)
    return images
image_list = load_images_from_folder("all_images/")
'''
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

image_list = load_images_from_folder("all_images/")
'''
#%% original image and scale
scale_percent = 60
width = int(img.shape[1]*scale_percent / 100)
height = int(img.shape[0]*scale_percent / 100)
dim =(width,height)

resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
cv2.imshow("orginal",resized)

resized_2 =cv2.resize(image_list[12],dim,interpolation = cv2.INTER_AREA)
cv2.imshow("resim 13",resized_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
def location_list(index):
    location_list = open("yolo_insaat_project/"+str(index)+".txt","r")
    list_of_locations=[]
    #print("orginal_txt/"+str(i)+".txt")
    for line in location_list:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        
        for number in line_list:
            n_list =float(number)
            print(n_list)
            list_of_locations.append(n_list)
    location_list.close()  
    
    return list_of_locations
    
       


#%% txt to list

list_of_images = []

for i in range(1,14):
    list_of_images.append(location_list(i))

#%% one to one xywh

def cal(index_1,index_2,index_3,index_4,dh,dw):
   x,y,w,h =int(index_1*dw),int(index_2*dh),int(index_3*dw),int(index_4*dh)
   list_one_to_one =[x,y,w,h]
   print(list_one_to_one)
   return list_one_to_one
#%% x,y,w h foncution

def x_y_w_h(list_of_location,dh,dw):
    
    list_all=[]
    for k in range(1,len(list_of_location)+1,5):
         
        if k+1>len(list_of_location):
            break
        
        list_all.append(cal(list_of_location[k],list_of_location[k+1],list_of_location[k+2],list_of_location[k+3],dh,dw))
    
    return list_all
#%%

list_all_xywh = []

for j in range(len(list_of_images)):
   
   list_all_xywh.append(x_y_w_h(list_of_images[j],dh,dw))
    

#%% crop single xywh

def crop_single_xywh(single_xywh,img,index_xywh,index_image):
    cropped_image = img[single_xywh[1]-5:single_xywh[1]+single_xywh[2]+5,single_xywh[0]-5:single_xywh[0]+single_xywh[3]-5] 
    cv2.imwrite("orjinal_resimler/orjinal_"+str(index_image+1)+"_"+str(index_xywh+1)+".jpg", cropped_image) 
#%% cropped image single

def crop_single_image(single_img_xywh,img,index_image): 
    for j in range(len(single_img_xywh)):
         crop_single_xywh(single_img_xywh[j],img,j,index_image)
 
  
#%% cropped image
def crop_image_all(list_of_xywh,imgs):
    
    for i in range(len(list_of_xywh)):
        crop_single_image(list_of_xywh[i],imgs[i],i)

#%%


crop_image_all(list_all_xywh,image_list)

#%%
cv2.imshow("resim 13",image_list[12])
cv2.waitKey(0)
cv2.destroyAllWindows()