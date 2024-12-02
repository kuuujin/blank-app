import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 모델 로드 함수
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

# 모델 초기화
model = load_model()

# Streamlit UI
st.title("Handwritten Digit Classification")
st.write("Upload a 28x28 grayscale image of a handwritten digit (0-9), and the model will classify it.")

# 사용자 이미지 업로드
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # 이미지 열기 및 처리
        image = Image.open(uploaded_file).convert("L")  # Grayscale로 변환
        image = ImageOps.invert(image)  # 색 반전 (흰 배경, 검은 글씨)
        image = image.resize((28, 28))  # 28x28 크기로 조정
        image_array = np.array(image) / 255.0  # 정규화

        # 데이터 형식 변환
        image_array = np.expand_dims(image_array, axis=-1)  # (28, 28, 1)로 변환
        image_array = np.expand_dims(image_array, axis=0)   # (1, 28, 28, 1)로 변환

        # 입력 데이터 크기 확인
        st.write(f"Processed input shape: {image_array.shape}")

        # 모델 예측
        prediction = model.predict(image_array, batch_size=1)
        predicted_label = np.argmax(prediction)

        # 결과 출력
        st.subheader("Prediction")
        st.write(f"The model predicts this digit as: **{predicted_label}**")

        # 예측 확률 시각화
        st.bar_chart(prediction.flatten())

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

else:
    st.write("Please upload an image to get started.")