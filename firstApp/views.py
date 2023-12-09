import os

from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage


# deep learning libraries
from keras.models import load_model
from keras.preprocessing import image
import json
import tensorflow as tf
import numpy as np

image_height = 224
image_width = 224

labels = "./models/imagenet_classes.json"

with open(labels, 'r') as f:
    label_info = f.read()

label_info = json.loads(label_info)

model = load_model("./models/MobileNetModelImageNet.h5")

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )


tf.compat.v1.disable_eager_execution()
model_graph = tf.compat.v1.get_default_graph()
tf_session = tf.compat.v1.Session()

with model_graph.as_default():
    with tf_session.as_default():
        model = load_model("./models/MobileNetModelImageNet.h5")


def index(request):
    # return HttpResponse("Hello")
    context = {'a': 1}
    return render(request, 'index.html', context)


def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()

    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName

    img = image.load_img(testimage, target_size=(image_height, image_width))
    x = image.img_to_array(img)
    x = x / 255
    x = x.reshape(1, image_height, image_width, 3)

    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    predictedLabel = label_info[str(np.argmax(predi[0]))]

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}

    return render(request, 'index.html', context)


def viewDataBase(request):
    listOfImages = os.listdir('./media')
    listOfImagesPath = ['./media/' + i for i in listOfImages]
    context = {"listOfImagesPath" : listOfImagesPath}
    return render(request, 'viewDB.html', context)
