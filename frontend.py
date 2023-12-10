import pickle
import streamlit as st
import cv2
import numpy as np
from PIL import Image

def calculate_tenengrad(gray_image):
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.sqrt(gradient_x**2 + gradient_y**2).mean()
    return tenengrad

def fungus_input_converter(image):
    # Here, need to reshape input image to numpy array
    # like how dataset is preprocessed, but with only one input  
    
    img_np = np.array(image)
    
    if img_np.ndim == 2: # image is already grayscale
        gray_img = img_np
    else: # converts RBG image to grayscale
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
    tenegrad = calculate_tenengrad(gray_img)

    return img_np, gray_img, tenegrad

def fungus_prediction(inputData):

    # Import model, run predictions on it    
    
    loaded_model = pickle.load(open('./model.sav', 'rb'))

    prediction = loaded_model.predict(inputData)
    
    if(prediction[0] == 1): # Assumes predictive algorithm returns 1, 2, 3, 5, or 6
        return 'This fungi is in class H1'
    elif(prediction[0] == 2): 
        return 'This fungi is in class H2'
    elif(prediction[0] == 3): 
        return 'This fungi is in class H3'
    elif(prediction[0] == 5):
        return 'This fungi is in class H5'
    elif(prediction[0] == 6):
        return 'This fungi is in class H6'
    else:
        return 'Classification error'

def main():
    
    st.title('Fungi classifier')
    
    inputImage = st.file_uploader('Upload a JPG/JPEG microscopic image of a fungal infection here:', type=['jpg'])
    
    if inputImage is not None:
        image = Image.open(inputImage)
        inputData, gray_image, tenegrad = fungus_input_converter(image)
        
        imgName = inputImage.name.split('.')[0] #gets rid of .jpg or .jpeg extension in name

        st.subheader(imgName)
        st.text('tenegrad value = ' + str(tenegrad))
        st.image([inputData, gray_image], caption=['Original Image', 'Grayscale Image'], width=300, channels='RBG')

    classification = ''
    
    if st.button('Classify!') and inputImage:
        classification = fungus_prediction(inputData)
        st.success(classification)
    


if __name__ == '__main__':
        main()

