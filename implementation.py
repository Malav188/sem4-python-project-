import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from keras import models
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


model = models.load_model("model.keras")

img = cv.imread("train_373.png ")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img,cmap=plt.cm.binary)
plt.show()

img = np.expand_dims(img, axis=0)
prediction = model.predict(np.array(img) / 255)

index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")
