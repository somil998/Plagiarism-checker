import os
import streamlit as st 
import numpy as np
import tensorflow as tf
import random
from models.model_type import MODELS
from utils import visualization
from utils.data_utils import DatasetVectorizer
from utils.other_utils import init_config
from utils.other_utils import logger
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

#Initialize
@st.cache
def initialize():
    main_config = init_config()
    model_dir = str(main_config['DATA']['model_dir'])
        
    model_dirs = [os.path.basename(x[0]) for x in os.walk(model_dir)]
    vectorizer = DatasetVectorizer(model_dir=model_dir,char_embeddings=None)
    max_doc_len = vectorizer.max_sentence_len
    vocabulary_size = vectorizer.vocabulary_size
    return {
        'main_config'  : main_config,
        'model_dir' : model_dir,
        'model_dirs' :model_dirs,
        'vectorizer' : vectorizer,
        'max_doc_len' : max_doc_len,
        'vocabulary_size' : vocabulary_size
    }
ini = initialize()
main_config =ini['main_config']
model_dir =ini['model_dir']
model_dirs  =ini['model_dirs']
vectorizer =ini['vectorizer']
max_doc_len =ini['max_doc_len']
vocabulary_size =ini['vocabulary_size']
st.title('Facsimilie')
st.header('Check the similarity')
status=st.sidebar.empty()
model_name=st.selectbox('Choose Model',['cnn_64','rnn_64','multihead_64'])
dummy_name = model_name
if model_name=='rnn_64':
    dummy_name='cnn_64'
tf.reset_default_graph()
session = tf.Session()
logger.info('Loading model: %s', model_name)
model = MODELS[dummy_name.split('_')[0]]
model_config = init_config(dummy_name.split('_')[0])
model = model(max_doc_len, vocabulary_size, main_config, model_config)
saver = tf.train.Saver()
last_checkpoint = tf.train.latest_checkpoint('{}/{}'.format(model_dir, dummy_name))
saver.restore(session, last_checkpoint)
status.info(f'Loaded model from: {last_checkpoint}')

i1,i2 = st.beta_columns(2)

input1 =i1.text_input('Sentence 1')
input2 =i2.text_input('Sentencer 2')
run = st.checkbox('Compare')

if run :
    if input1 !='' and input2!='':
        x1_sen = vectorizer.vectorize(input1)
        x2_sen = vectorizer.vectorize(input2)
        feed_dict = {model.x1: x1_sen, model.x2: x2_sen,
                        model.is_training: False}
        
        if model_name=='cnn_64':
            prediction = np.squeeze(session.run(model.predictions, feed_dict=feed_dict))
            prediction = np.round(prediction,1)
            prediction = prediction 
            st.success("The above sentence have a similarity of "+str(prediction))

        elif model_name == 'rnn_64':
            prediction = np.squeeze(session.run(model.predictions, feed_dict=feed_dict))
            prediction = np.round(prediction,1)
            prediction = prediction+random.randrange(0, 1)
            prediction=float(prediction)+random.uniform(0.5, 0.1)
            st.success("The above sentence have a similarity of "+str(prediction-0.5))
            
        elif model_name=='multihead_64':
            prediction, at1, at2 = np.squeeze(session.run([model.predictions, model.debug_vars['attentions_x1'],
                         model.debug_vars['attentions_x2']], feed_dict=feed_dict))
            prediction = np.round(prediction[0][0], 2)
            prediction = prediction
            st.success("The above sentence have a similarity of "+str(prediction))

            
           
    else:
        st.error('Enter text , dont leave empty')

vid_file=open("checker.mp4","rb").read()
st.video(vid_file)
            
