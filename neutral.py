import os
import cv2
import numpy as np

path = "dataset/neutral"   # 👈 neutral folder
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
        labels.append(4)   # 👈 label for neutral

    except Exception as e:
        print("Error:", img)

data = np.array(data)
labels = np.array(labels)

print("Neutral data loaded:", data.shape)