import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
from PIL import Image

# AWS S3 configuration
bucket_name = "textile-inspection"  # Replace with your S3 bucket name
image_key = "images/W001.jpg"    # Replace with your S3 object key

# Initialize S3 client
s3 = boto3.client("s3")

# Download image from S3
response = s3.get_object(Bucket=bucket_name, Key=image_key)
image_data = response["Body"].read()

# Convert the downloaded image to a NumPy array
image = Image.open(BytesIO(image_data)).convert("L")  # Convert to grayscale
source_image = np.array(image)

# Get image shape
image_height, image_width = source_image.shape

# Sliding window parameters
window_size = 160  
window_stride = 1  
ifv_step = 20
ifv_width = 10

ifv_dx = []
ifv_dy = []

for y in range(0, image_height - window_size + 1, window_stride):
    for x in range(0, image_width - window_size + 1, window_stride):
        window = source_image[y:y + window_size, x:x + window_size]
        for fr in range(0, int(window_size / 2) - ifv_width, ifv_step):
            print(fr)
            ifv_filter = np.zeros((window_size, window_size))
            ifv_filter[fr:window_size-fr, fr:window_size-fr] = 1
            
            ifv_filter[fr+ifv_width:window_size-(fr+ifv_width), fr+ifv_width:window_size-(fr+ifv_width)] = 0
            ifwin = ifv_filter * window
            
            ifwin_dir = np.array(ndimage.measurements.center_of_mass(ifwin))
            dx = ifwin_dir[1] - (window_size / 2)
            dy = ifwin_dir[0] - (window_size / 2)
            
            ifv_dx.append(dx / (window_size / 2))
            ifv_dy.append(dy / (window_size / 2))
            
            fig = plt.figure()
            ax0 = fig.add_subplot(1, 3, 1)
            ax0.imshow(window, origin='lower', interpolation='None', cmap='gray', alpha=0.6)
            
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(ifv_filter, origin='lower', interpolation='None', cmap='gray', alpha=0.6)
            
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(ifwin, origin='lower', interpolation='None', cmap='gray', alpha=0.6)
            ax3.arrow(((window_size - 1) / 2), ((window_size - 1) / 2), dx, dy, head_width=0.05, head_length=0.1, fc='g', ec='k')
        break
    break
plt.show()
