import streamlit as st
import cv2 as cv
import numpy as np
from detection_model import DetectionModel

if __name__=='__main__':
    model_path = 'detection_model.pt'
    model = DetectionModel(model_path)

    st.title('Kidney Stones Detection')
    st.write('Kidney stones (also called renal calculi, nephrolithiasis or urolithiasis) are hard deposits made of minerals and salts that form inside your kidneys. Kidney stones vary in density, but most are composed of substances like calcium oxalate (and sometimes calcium phosphate). These mineral components have a high density, resulting in their bright appearance on the scan.')

    file=st.file_uploader('',type=["jpg", "jpeg", "png"])

    if file is not None:
        image = cv.imdecode(np.fromstring(file.read(), np.uint8), cv.IMREAD_COLOR)

        mod_img = model.predict(image)

        col1,col2=st.columns(2)
        col1.image(image, channels="BGR", caption="Uploaded Image.", use_column_width=True)
        col2.image(mod_img, caption="Processed Image.", use_column_width=True)
