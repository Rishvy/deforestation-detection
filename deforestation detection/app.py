import streamlit as st
import pandas as pd
import numpy as np
import rasterio 
import pickle
import matplotlib.pyplot as plt
import os
from PIL import Image
import tempfile
import io
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns

st.title("Deforestation Detection - WebApp")
st.markdown("Deforestation entails the extensive removal of trees, primarily due to human activities like logging and agriculture, this cause ecological imbalance and climate repercussions. Leverage this advanced model to swiftly determine the existing condition of a specified area – whether it has undergone deforestation or remains unaffected")

st.sidebar.title("Upload ⬆️ ")
st.sidebar.markdown("Upload NDVI satellite image of the area that you want to classify")
st.sidebar.markdown("(Only TIF format images are accepted)")

col1, col2 = st.columns(2)

with open('deforestation_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

    X_test = np.load(r'C:\Users\Rishvy\Desktop\Detecting_deforestation_using_satellite_images-master\Data\X_test.npy')
    y_test = np.load(r'C:\Users\Rishvy\Desktop\Detecting_deforestation_using_satellite_images-master\Data\y_test.npy')
 
y_pred = loaded_model.predict(X_test)

class_names = ['Non-Deforested','Deforested']

@st.cache_resource
def prediction(mean_ndvi_value):
    file_name = 'deforestation_model.pkl'  # Corrected file name
    with open(file_name, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([[mean_ndvi_value]])
    return pred_value[0] 

def calculate_mean_ndvi2(tif_path):
    with rasterio.open(tif_path) as src:
        ndvi_data = src.read(1)
        valid_ndvi_values = ndvi_data[~np.isnan(ndvi_data)]
        mean_ndvi = np.mean(valid_ndvi_values)

    return mean_ndvi

def upload_image():
    my_upload = st.sidebar.file_uploader("Upload an image", type=["tif"])
    if my_upload is not None:
        return my_upload
    else:
        st.write("Upload an image to visualize.")

def visualize_tiff_images(upload):
    temp_file_name = None
    if upload is not None:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
            temp_file.write(upload.read())
            temp_file_name = temp_file.name
            temp_file.close()
        with rasterio.open(temp_file_name) as src:
            ndvi_data = src.read(1)
            cmap = plt.cm.RdYlGn
            cmap.set_bad(color='brown')
            img_array = cmap(ndvi_data)
            rgb_img_array = (255 * img_array[:, :, :3]).astype(np.uint8)
            img = Image.fromarray(rgb_img_array)
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format="JPEG")
            jpeg_img = jpeg_buffer.getvalue()
            col1.image(jpeg_img, caption="Uploaded NDVI Image", use_column_width=False, width=300)
            agree = st.sidebar.checkbox('Mark Deforested Area')
            if agree:
                threshold = 0.33
                deforested_mask = ndvi_data < threshold
                ndvi_data[deforested_mask] = np.nan
                cmap = plt.cm.RdYlGn
                cmap.set_bad(color='red')
                img_array = cmap(ndvi_data)
                rgb_img_array = (255 * img_array[:, :, :3]).astype(np.uint8)
                img = Image.fromarray(rgb_img_array)
                jpeg_buffer = io.BytesIO()
                img.save(jpeg_buffer, format="JPEG")
                jpeg_img = jpeg_buffer.getvalue()
                col2.image(jpeg_img, caption="Deforested Areas", use_column_width=False, width=300)
                return temp_file_name
            return temp_file_name
    else:
        st.sidebar.text("Upload a TIFF image to visualize.")
    return temp_file_name

def show_accuracy():
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.write(accuracy)

def show_precision():
    precision = precision_score(y_test, y_pred)
    st.subheader("Precision")
    st.write(precision)

def show_confusion_matrix():
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    st.pyplot()


def main():
    uploaded_image =  upload_image()
    if uploaded_image is not None:
        input_img = visualize_tiff_images(uploaded_image)

    if uploaded_image is not None:
        st.sidebar.title("Results")
        if st.sidebar.button('Deforestation Status of Area'):
            mean_ndvi = calculate_mean_ndvi2(input_img)
            st.write("Mean NDVI Value:", mean_ndvi)
            prediction_result = prediction(mean_ndvi)
            if prediction_result == 1:
                st.write("The area is classified as Deforested Area.")
            else:
                st.write("The area is classified as non-deforested Area.")

    st.sidebar.title("Model Performance Matrices")
    option = st.sidebar.selectbox(
    'Select a performance measure',
    ('-', 'Accuracy', 'Precision', 'Confusion Matrix'))

    options_dict = {
    'Accuracy': show_accuracy,
    'Precision': show_precision,
    'Confusion Matrix': show_confusion_matrix,
    }
    
    if option in options_dict:
        options_dict[option]()

if __name__ == '__main__':
    main()
