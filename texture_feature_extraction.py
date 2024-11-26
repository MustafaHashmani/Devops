# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

import time
import cv2
import numpy as np
import csv

from scipy import ndimage


win_size=160
win_hop=80

# Texture Feature List
featureFile=open('features_ws'+str(win_size)+'_wh'+str(win_hop)+'.csv', 'w',newline='')
featureFileWriter = csv.writer(featureFile,delimiter=';') 
line=["Image","h","w","t"]

for n in ["Cont_","Diss_","Homo_","Ener_","Corr_"]:
        for angle in [0,45,90]:
            line.append(n+"_"+str(angle))
            #line.append(n+"_"+str(angle)+"_ET")

for lbl in ["20","40","80"]:
    line.append("LBP_"+lbl)
    #line.append("lbp "+lbl+" ET")

for theta in np.arange(0,np.pi,np.pi/8):
    for sigma in (1, 3 ,5):
        for frequency in [0.1, 0.3, 0.5, 1, 3, 5 ]:
            line.append("GFB_M_"+str(theta)+"_"+str(sigma)+"_"+str(frequency))
            line.append("GFB_V_"+str(theta)+"_"+str(sigma)+"_"+str(frequency))
            line.append("GFB_PP_"+str(theta)+"_"+str(sigma)+"_"+str(frequency))

ifv_step=20
ifv_width=10
for fr in range(0,int(win_size/2)-ifv_width, ifv_step):            
    line.append("ifv_dx_"+str(fr)+"_"+str(ifv_width))
    line.append("ifv_dy_"+str(fr)+"_"+str(ifv_width))

# Write the coulmn names
featureFileWriter.writerow(line)
linecount=0

# Gabor Filter Bank Feature Extractor
def gaborfilterBankfeats(image):
    btime=time.time()
    kernels = []
    for theta in np.arange(0,np.pi,np.pi/8):
        for sigma in (1, 3 ,5):
            for frequency in [0.1, 0.3, 0.5, 1, 3, 5 ]:
                kernel = np.real(gabor_kernel(frequency, theta=theta* np.pi,sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    feats = np.zeros((len(kernels), 3), dtype=np.double)
    
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        feats[k, 2] = filtered.max()-filtered.min()
    et=btime-time.time()
    return [feats,et]

def IFV(source_image,window_size,ifv_step,ifv_width):
    image_height, image_width = source_image.shape
    ifv_dx = []
    ifv_dy = []
    for fr in range(0,int(window_size/2)-ifv_width,ifv_step):
        ifv_filter=np.zeros((window_size,window_size))
        ifv_filter[fr:window_size-fr,fr:window_size-fr]=1
        #IFV filtering
        ifv_filter[fr+ifv_width:window_size-(fr+ifv_width),fr+ifv_width:window_size-(fr+ifv_width)]=0
        ifwin=ifv_filter*source_image
        #IFV calculation
        ifwin_dir=np.array(ndimage.measurements.center_of_mass(ifwin))
        dx=ifwin_dir[1]-(window_size/2)
        dy=ifwin_dir[0]-(window_size/2)
        #normalization
        ifv_dx.append(dx/(window_size/2))
        ifv_dy.append(dy/(window_size/2))
    return ifv_dx, ifv_dy

## Texture Feature Extraction
for i in range(1,131):
    filename="W"+str(i).zfill(3)+".jpg"
    source_image = cv2.imread("images/"+filename,0)
    
    height = source_image.shape[0]
    width = source_image.shape[1]
    
    half_win = win_size // 2

    for y in range(half_win, height - half_win, win_hop):
        for x in range(half_win, width - half_win, win_hop):
            linecount=linecount+1
            print("###############################################")
            print(linecount,"File:"+filename, "x:",x,"y:",y)
            line=[filename, str(y),str(x),str(i)]
            
            sw= (y - half_win)+ (y + half_win + 1)
            sh= (x - half_win)+ (x + half_win + 1)
            window = source_image[y - half_win:y + half_win , x - half_win:x + half_win ]
            
            """
            plt.close()
            plt.subplot(2,2,1)
        
            plt.imshow(source_image)
            ax = plt.gca()
            rect = patches.Rectangle((x,y),win_size,win_size,linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            plt.subplot(2,2,2)
            plt.imshow(window)
        
            plt.show()
            plt.waitforbuttonpress()
            
            """
            #GLCM http://matlab.izmiran.ru/help/toolbox/images/enhanc15.html
            glcm = graycomatrix(window, [1], [0, 0.25*np.pi, 0.5*np.pi], 256, symmetric=True, normed=True)
            for feature in ["contrast","dissimilarity","homogeneity","energy","correlation"]:
                btime=time.time()
                fang=graycoprops(glcm, feature)
                #print("glcm",feature, fang)
                for fa in fang[0]:
                    line.append(str(fa).replace(".",","))
                    et=btime-time.time()
                    #line.append("et:"+str(et).replace(".",","))
        
            #LBP https://liris.cnrs.fr/Documents/Liris-5004.pdf
            btime=time.time()
            lbp80 = local_binary_pattern(window, 20, win_size/2, 'uniform')
            line.append(str(np.sum(lbp80)).replace(".",","))
            et=btime-time.time()
            #line.append("et:"+str(et).replace(".",","))
            
            btime=time.time()
            lbp80 = local_binary_pattern(window, 40, win_size/2, 'uniform')
            line.append(str(np.sum(lbp80)).replace(".",","))
            et=btime-time.time()
            #line.append("et:"+str(et).replace(".",","))
            
            btime=time.time()
            lbp160 = local_binary_pattern(window, 80, win_size/2, 'uniform')
            line.append(str(np.sum(lbp160)).replace(".",","))
            et=btime-time.time()
            #line.append("et:"+str(et).replace(".",","))
            #print("lbp4:",et)
        
            # gabor filter bank
            fb_feat,et=gaborfilterBankfeats(window)
            for gff in fb_feat:
                line.append(str(gff[0]).replace(".",","))
                line.append(str(gff[1]).replace(".",","))
                line.append(str(gff[2]).replace(".",","))
            #line.append("et:"+str(et/144).replace(".",","))
            
            # IFV
            ifv_step=20
            ifv_width=10
            ifv_dx,ifv_dy=IFV(window, win_size ,ifv_step,ifv_width)
            for dx in ifv_dx:
                line.append(str(dx).replace(".",","))
            for dy in ifv_dy:
                line.append(str(dx).replace(".",","))
            featureFileWriter.writerow(line)
featureFile.close()
           