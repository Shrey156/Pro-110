import cv2
import numpy as np
import tensorflow as tf

model=tf.keras.models.load_model("keras_model.h5")

img = cv2.VideoCapture(0)

while True:
    success,frame = img.read()
    
    #resize the img
    new_img=cv2.resize(frame,(224,224))
    
    # convert the img into numpy array and increase dimension
    test_img=np.array(new_img,dtype=np.float32)
    test_img=np.expand_dims(test_img,axis=0)
    
    # normalising the image
    nor_img=test_img/255.0
    
    prediction = model.predict(nor_img)
    print(prediction)
    cv2.imshow("result",frame)
    key = cv2.WaitKey(1)
    if key ==32:
        break

img.release()    
