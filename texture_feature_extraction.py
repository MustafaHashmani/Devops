# -*- coding: utf-8 -*-
import boto3
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
from io import BytesIO

# AWS S3 configuration
s3 = boto3.client('s3')
bucket_name = 'textile-inspection'
image_prefix = 'images/'
output_file_name = f'features_ws160_wh80.csv'

# Download an image from S3
def download_image_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return cv2.imdecode(np.frombuffer(response['Body'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# Upload the CSV file to S3
def upload_csv_to_s3(bucket, key, file_path):
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded {file_path} to s3://{bucket}/{key}")

# Define constants
win_size = 160
win_hop = 80

# Create the CSV file locally before uploading
local_csv_path = output_file_name
featureFile = open(local_csv_path, 'w', newline='')
featureFileWriter = csv.writer(featureFile, delimiter=';')
line = ["Image", "h", "w", "t"]

for n in ["Cont_", "Diss_", "Homo_", "Ener_", "Corr_"]:
    for angle in [0, 45, 90]:
        line.append(n + "_" + str(angle))

for lbl in ["20", "40", "80"]:
    line.append("LBP_" + lbl)

for theta in np.arange(0, np.pi, np.pi/8):
    for sigma in (1, 3, 5):
        for frequency in [0.1, 0.3, 0.5, 1, 3, 5]:
            line.append("GFB_M_" + str(theta) + "_" + str(sigma) + "_" + str(frequency))
            line.append("GFB_V_" + str(theta) + "_" + str(sigma) + "_" + str(frequency))
            line.append("GFB_PP_" + str(theta) + "_" + str(sigma) + "_" + str(frequency))

ifv_step = 20
ifv_width = 10
for fr in range(0, int(win_size/2)-ifv_width, ifv_step):
    line.append("ifv_dx_" + str(fr) + "_" + str(ifv_width))
    line.append("ifv_dy_" + str(fr) + "_" + str(ifv_width))

# Write the column names
featureFileWriter.writerow(line)
linecount = 0

# Gabor Filter Bank Feature Extractor
def gaborfilterBankfeats(image):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi/8):
        for sigma in (1, 3, 5):
            for frequency in [0.1, 0.3, 0.5, 1, 3, 5]:
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    feats = np.zeros((len(kernels), 3), dtype=np.double)

    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        feats[k, 2] = filtered.max() - filtered.min()

    return feats

# IFV feature extraction
def IFV(source_image, window_size, ifv_step, ifv_width):
    image_height, image_width = source_image.shape
    ifv_dx = []
    ifv_dy = []
    for fr in range(0, int(window_size/2)-ifv_width, ifv_step):
        ifv_filter = np.zeros((window_size, window_size))
        ifv_filter[fr:window_size-fr, fr:window_size-fr] = 1
        ifv_filter[fr+ifv_width:window_size-(fr+ifv_width), fr+ifv_width:window_size-(fr+ifv_width)] = 0
        ifwin = ifv_filter * source_image
        ifwin_dir = np.array(ndimage.measurements.center_of_mass(ifwin))
        dx = ifwin_dir[1] - (window_size/2)
        dy = ifwin_dir[0] - (window_size/2)
        ifv_dx.append(dx / (window_size/2))
        ifv_dy.append(dy / (window_size/2))
    return ifv_dx, ifv_dy

# Process each image in S3
for i in range(1, 131):
    filename = f"W{str(i).zfill(3)}.jpg"
    s3_key = f"{image_prefix}{filename}"

    source_image = download_image_from_s3(bucket_name, s3_key)
    height, width = source_image.shape

    half_win = win_size // 2
    for y in range(half_win, height - half_win, win_hop):
        for x in range(half_win, width - half_win, win_hop):
            linecount += 1
            print("###############################################")
            print(linecount, "File:", filename, "x:", x, "y:", y)
            line = [filename, str(y), str(x), str(i)]

            window = source_image[y - half_win:y + half_win, x - half_win:x + half_win]

            # GLCM
            glcm = graycomatrix(window, [1], [0, 0.25 * np.pi, 0.5 * np.pi], 256, symmetric=True, normed=True)
            for feature in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
                fang = graycoprops(glcm, feature)
                for fa in fang[0]:
                    line.append(str(fa).replace(".", ","))

            # LBP
            lbp = local_binary_pattern(window, 20, win_size/2, 'uniform')
            line.append(str(np.sum(lbp)).replace(".", ","))

            # Gabor Filter Bank
            fb_feat = gaborfilterBankfeats(window)
            for gff in fb_feat:
                line.append(str(gff[0]).replace(".", ","))
                line.append(str(gff[1]).replace(".", ","))
                line.append(str(gff[2]).replace(".", ","))

            # IFV
            ifv_dx, ifv_dy = IFV(window, win_size, ifv_step, ifv_width)
            for dx in ifv_dx:
                line.append(str(dx).replace(".", ","))
            for dy in ifv_dy:
                line.append(str(dy).replace(".", ","))

            featureFileWriter.writerow(line)

# Close CSV file and upload to S3
featureFile.close()
upload_csv_to_s3(bucket_name, f"output/{output_file_name}", local_csv_path)
