import os
import numpy as np
from keras.models import load_model
import cv2
import sys
import config

# Dinh nghia class
# image_size = 224
# class_names = ['banana_lv1','banana_lv2','banana_lv3']

# Load model da train
my_model = load_model(os.path.join("models","banana_model.h5"))

# Doc anh từ tham số dòng lệnh
image_org = cv2.imread(sys.argv[1])

# Resize
image = image_org.copy()
image = cv2.resize(image, dsize=(config.image_size, config.image_size))

# Convert to tensor
image = np.expand_dims(image, axis=0)

# Dự đoán
predict = my_model.predict(image)
print("This picture is: ", config.class_names[np.argmax(predict)])

# Show image
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

cv2.putText(image_org, config.class_names[np.argmax(predict)] , org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("Picture", cv2.resize(image_org, dsize=None, fx=0.3, fy=0.3))
cv2.waitKey()
