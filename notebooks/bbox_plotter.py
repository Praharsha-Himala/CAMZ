import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt

'''Author : nvs - version 2.0'''

def midpt_to_corners(box):

    center_x,center_y,width,height = box

    # Convert YOLO format (center_x, center_y, w, h) to (x1, y1, x2, y2)
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)
    return [x1,y1,x2,y2]

def bbox_calculator(image, txt_file_path):
        bbox = []
        image_height, image_width, _ = image.shape
        with open(txt_file_path, 'r') as file:
                for line in file:
                # Parse the YOLO annotation (class_id, center_x, center_y, width, height)
                    data = line.strip().split()
                    center_x = float(data[1]) * image_width
                    center_y = float(data[2]) * image_height
                    width = float(data[3]) * image_width
                    height = float(data[4]) * image_height
                    bbox.append(midpt_to_corners([center_x,center_y,width,height]))
        return bbox

def plot_bbox(image,label):
    """This function shows the bounding box on the image

    Args:
        image(str or arraylike): 
            str : images folder
            arraylike : image as an array
        label(str or arraylike):
            str : labels folder containing the .txt file as label in YOLO format
            arraylike : bounding box of shape (1,4)
    """
         
    if (type(label)==str):
        files = os.listdir(image)
        images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random_image = random.choice(images)
        random_image_path = os.path.join(image, random_image)
        image = cv2.imread(random_image_path)
        txt_file = os.path.splitext(random_image)[0] + '.txt'
        txt_file_path = os.path.join(label, txt_file)
        bbox = bbox_calculator(image, txt_file_path)
    else:
        if (type(image) == str) : 
            image = cv2.imread(image)
        else : 
            image = np.asarray(image)
        bbox = label
    
    bbox = np.asarray(bbox)
    x1, y1, x2, y2 = bbox[0]
    color = (0, 0, 255) 
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    plt.axis('off')
    plt.imshow(image);           
    return 