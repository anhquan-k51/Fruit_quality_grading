from keras.models import load_model
from flask import Flask, render_template, request
import os
from random import random
import cv2
import numpy as np
import config

# Khởi tạo Flask
app = Flask(__name__)

# image_size = 224
# class_names = ['banana_lv1','banana_lv2','banana_lv3']

# Load model da train
my_model = load_model(os.path.join("models", "banana_model.h5"))


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join("static", image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Đọc ảnh và resize về kích thước input model
                frame = cv2.imread(path_to_save)
                frame = cv2.resize(frame, dsize=(config.image_size, config.image_size))
                # Convert to tensor
                frame = np.expand_dims(frame, axis=0)

                # Dự đoán bằng model
                predict = my_model.predict(frame)

                # Lấy tên của class
                predict_name = config.class_names[np.argmax(predict)]

                # If then để in ra câu thông báo thêm cho vui nhộn
                if predict_name.endswith("2"):
                    extra = "Chuối ăn được rồi bạn ơi!"
                elif predict_name.endswith("1"):
                    extra = "Chuối còn non và xanh lắm!"
                else:
                    extra = "Chuối chín quá rồi bạn nên bỏ đi !"

                # Trả về kết quả
                return render_template("index.html", user_image = image.filename , rand = str(random()),
                                       msg="Tải file lên thành công",
                                       fruit_level= predict_name, extra=extra)

            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được ảnh')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, use_reloader=False)