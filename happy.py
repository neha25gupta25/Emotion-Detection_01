import os
import cv2
import numpy as np

path = "dataset/happy"   # only happy folder
IMG_SIZE = 48

data = []
labels = []

for img in os.listdir(path):
    try:
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        data.append(image)
        labels.append(0)   # label for happy

    except Exception as e:
        print("Error:", img)

data = np.array(data)
labels = np.array(labels)

print("Happy data loaded:", data.shape)