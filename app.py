import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt 
import time
from predictionOnImages import predict_output
fig = plt.figure()

st.title("Blood Cancer Detection")
uploaded_file = st.file_uploader("Upload a File")
btn_classify = st.button("Classify")
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption = 'Uploaded Image', use_column_width=True)
if btn_classify:
  if uploaded_file is None:
    st.write("Image not found! Please upload an image.")
  else:
    with st.spinner("Model working....."):
      plt.imshow(image)
      plt.axis("off")
      time.sleep(1)
      st.success("Classified")
      st.write(predict_output(image))
