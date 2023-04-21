from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import keras.utils as image   
from keras.applications.vgg16 import preprocess_input

model_json = None

with open("./model/self_trained/BLOODCANCER_model.json", "r") as file:
    model_json = file.read()
loaded_model = model_from_json(model_json)
# loaded_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
loaded_model.load_weights('model/self_trained/model_2.h5')

activity_label = {'0': 'Healthy',
                  '1': 'Acute Lukemia',
                  }

#{'c3': 0, 'c7': 1, 'c0': 2, 'c2': 3, 'c9': 4, 'c1': 5, 'c4': 6, 'c5': 7, 'c6': 8, 'c8': 9}
# 5 - Drinking
#6 - c0
#5 - c1, c6, c2, 
#Using phone
labels = {
    0: "Healthy",
    1: "Acute Leukemia"  
}

def change_format_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    # img = image.load_img(img_path, target_size=(128, 128))
    img = np.asarray(img_path)
    img = cv2.resize(img, dsize=(300, 300))
    # convert PIL.Image.Image type to 3D tensor with shape (128, 128, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 128, 128, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def predict_output(file):
    file = change_format_to_tensor(file).astype('float32')/255 
    result = loaded_model.predict(file)
    
    return labels[np.argmax(result)]
    # print(np.argmax(result))
    # for value  in result:
    #     print(value)
    # for i in range(10):
    #     if maxVal < result[i]:
    #         index = i
    #         maxVal = result[i]
    
