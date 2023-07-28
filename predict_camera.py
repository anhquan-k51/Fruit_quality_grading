import os
import numpy as np
from keras.models import load_model
import cv2
import sys
import config
from imutils.video import VideoStream

# Dinh nghia class
# image_size = 224
# class_names = ['banana_lv1','banana_lv2','banana_lv3']

# Load model da train
my_model = load_model(os.path.join("models", "banana_model.h5"))

video = VideoStream(src=0).start()
# Show image
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

# Load webcam
while True:
    image_org = video.read()
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(config.image_size, config.image_size))

    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Dự đoán
    predict = my_model.predict(image)
    if np.max(predict)>0.5:
        print("Độ tin cậy nhận diện = ", np.max(predict))
        print("This picture is: ", config.class_names[np.argmax(predict)])

        cv2.putText(image_org, config.class_names[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1)==ord("q"):
        break

video.stop()
cv2.destroyAllWindows()