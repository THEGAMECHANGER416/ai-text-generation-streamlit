import numpy as np
import tensorflow as tf
import streamlit as st
import tensorflow_text as tf_text

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/saved_model.pb')
    return model

model=load_model()


st.write("# Chat-GPT Essay Detection Model 🤫")
label = "## Write or copy-paste any piece of text and this model will tell you the probability of the text being written by an AI"
text = st.text_area(label)
if st.button(label='Submit'):
    if text is not None:
        res = model.predict(np.array([text]))
        if res[0][0] < 0.5:
            st.write(f"### It is HUMAN GENERATED🙃! \n Probability of it being human is:{(1 - res[0][0]) * 100:.2f}%")
        else:
            st.write(
                f"### This was written by an AI 🤖! \n Probability of it being written by AI is:{(res[0][0]) * 100:.2f}%")
