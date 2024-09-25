import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input


IMAGE_ADDRESS = "https://smarttek.solutions/wp-content/uploads/ai-in-agriculture-1170x516.jpg"
IMG_SIZE = (224, 224)
IMAGE_NAME = "apple_disease.png"
MODEL_NAME = "mlp_final_model"

PREDICTION_LABELS = [
    "Complex","Frog Eye Leaf Spot","Healthy","Powdery Mildew","Rust", "Scab" ]
PREDICTION_LABELS.sort()


# functions
@st.cache_resource
def get_convext_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtXLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions


# get the featurization model
featurized_model = get_convext_model()
# load Portrait model
disease_model = load_sklearn_models(MODEL_NAME)


#Building the website

#title of the web page
st.title("Apple Leaf Disease Classification")

#setting the main picture
st.image(IMAGE_ADDRESS, caption = "AI in Agriculture")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.markdown("**This web app is about predicting whether an given apple plant has a disease or not.**")
    st.markdown("**Categories the web app can predict,**")
    for pred_class in PREDICTION_LABELS:
        st.markdown("- :red[{}]".format(pred_class))


#setting file uploader
#you can change the label name as your preference
# File uploader
image = st.file_uploader("Uplaod a picture of the plant...", type=['jpg','jpeg','png'])

if image:
  #displaying the image
  st.image(image, caption = "User Uploaded Image")
  user_image = Image.open(image)
  # save the image to set the path
  user_image.save(IMAGE_NAME)

  #get the features
  with st.spinner("Processing......."):
    image_features = featurization(IMAGE_NAME, featurized_model)
    model_predict = disease_model.predict(image_features)
    model_predict_proba = disease_model.predict_proba(image_features)
    probability = model_predict_proba[0][model_predict[0]]
  col1, col2 = st.columns(2)

  with col1:
    st.header("Disease Condition")
    st.subheader("{}".format(PREDICTION_LABELS[model_predict[0]]))
  with col2:
    st.header("Prediction Probability")
    st.subheader("{}".format(probability))