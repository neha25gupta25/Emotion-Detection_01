import os
import cv2
import numpy as np

path = "dataset/surprised"   # 👈 surprised folder
IMG_SIZE = 48

data = []
labels = []

for img in os.listdir(path):
    img_path = os.path.join(path, img)

    try:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        data.append(image)
        labels.append(3)   # 👈 label for surprised

    except Exception as e:
        print("Error:", img)

data = np.array(data)
labels = np.array(labels)

print("Surprised data loaded:", data.shape)