import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Import mixed precision từ Keras để sử dụng huấn luyện mixed-precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Tải bộ dữ liệu MNIST và chia thành tập huấn luyện và tập kiểm tra
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu để đưa giá trị pixel vào khoảng từ 0 đến 1
x_train = x_train / 255
x_test = x_test / 255

# Thêm một chiều kênh vào ảnh (được yêu cầu đối với CNN)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Chuyển đổi dữ liệu thành định dạng nhị phân (0 hoặc 1) để sử dụng trong web app
threshold = 0.5
x_train = (x_train > threshold).astype(np.int8)
x_test = (x_test > threshold).astype(np.int8)

# Tạo mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 lớp cho các chữ số từ 0 đến 9
])

# Biên dịch mô hình với hàm mất mát và bộ tối ưu phù hợp
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình trong 5 epoch và kiểm tra trên tập kiểm tra
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Lưu mô hình đã huấn luyện dưới định dạng TensorFlow SavedModel để chuyển đổi sang TensorFlowJS
tf.saved_model.save(model, "mnist_model")
