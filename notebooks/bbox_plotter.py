import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

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

    
def midpt_to_corners(labels, image_width, image_height, labeldim=2):
    
    if labeldim == 1:
        
        center_x = labels[0] * image_width
        center_y = labels[1] * image_height
        box_width = labels[2] * image_width
        box_height = labels[3] * image_height

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        return torch.tensor([x1, y1, x2, y2])
    
    if labeldim==2:
        center_x = labels[:, 0] * image_width
        center_y = labels[:, 1] * image_height
        box_width = labels[:, 2] * image_width
        box_height = labels[:, 3] * image_height

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2

        return torch.stack([x1, y1, x2, y2], dim=1)

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
        if (type(image) == str): 
            image = cv2.imread(image)
        else : 
            image = np.asarray(image)
        bbox = label
    
    bbox = np.asarray(bbox)
    if bbox.ndim > 1:
        x1, y1, x2, y2 = bbox[0]
        color = (0, 0, 255)
    else:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
     
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    plt.axis('off')
    plt.imshow(image);           
    return


def visualize_prediction(image_path, labels_path, model):
    image = cv2.imread(image_path)
    orig_image = image.copy()
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (224, 224))
    input_tensor = torch.tensor(resized_image).permute(2, 0, 1).unsqueeze(0).float()  # (B, C, H, W)
    input_tensor = input_tensor / 255.0
    input_tensor = input_tensor.to('cpu')

    # Get predictions from the model
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor).squeeze(0)
    predictions_norm = midpt_to_corners(predictions, image_height=height, image_width=width, labeldim=1)

    # Check if corresponding label file exists
    if labels_path:
        # Extract the base name of the image file without extension
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_file_path = os.path.join(labels_path, label_name)

        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                original_labels = []
                for line in file:
                    class_id, center_x, center_y, w, h = map(float, line.strip().split())
                    original_labels.append([center_x, center_y, w, h])
            original_labels = np.array(original_labels)
            original_bboxes = midpt_to_corners(torch.tensor(original_labels), image_width=width, image_height=height, labeldim=2)
            
            for bbox in original_bboxes:
                x1, y1, x2, y2 = map(int, bbox.tolist())
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for original labels

    # Plot predictions
    x1, y1, x2, y2 = map(int, predictions_norm.tolist())
    cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for predictions

    # Display the image
    plt.axis('off')
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.show()
    