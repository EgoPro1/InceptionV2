from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2             
from tensorflow.python.keras.preprocessing import image as imgx                                          
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

model = None


def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    model1=InceptionResNetV2(weights='imagenet')
    if model1 is None:
        print("Model1 NOT loaded")
    else:
        print("Model1  loaded")
    print("Model loaded")    
    return model1


def predict(img: Image.Image):
    global model
    if model is None:
        model = load_model()
    print("imagen ")
    print(img)
    img = np.asarray(img.resize((299, 299)))[..., :3]
    x = imgx.img_to_array(img)                                                      
    x = np.expand_dims(x, axis=0)                                                    
    x = preprocess_input(x)                                                          
                                                                                 
    preds = model.predict(x)                                                         
    print ('Prediction:', decode_predictions(preds, top=1)[0][0])
    res= decode_predictions(preds, top=1)[0][0]

    response = []
    
    resp = {}
    resp["class"] = res[1]
    resp["confidence"] = f"{res[2]*100:0.2f} %"

    response.append(resp)

    return response


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
