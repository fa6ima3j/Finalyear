#Inspiration
#https://www.#youtube.com/watch?v=3EBdT-0gvu8
#https://github.com/joeVenner/FaceRecognition-GUI-APP/blob/master/Detector.py

#Modified code below:

import numpy as np
from PIL import Image
import os, cv2



# Method to train custom classifier to recognize bag
def train_classifer(style):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd()+"/data/"+style+"/")

    bags = []
    ids = []
    pictures = {}


    # Store images in a numpy format and ids of the styles on the same index in imageNp and id lists

    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures :

            imgpath = path+pic
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(pic.split(style)[0])
            #names[style].append(id)
            bags.append(imageNp)
            ids.append(id)

    ids = np.array(ids)

    #Train and save classifier
    clf = cv2.bag.LBPHRecognizer_create()
    clf.train(bags, ids)
    clf.write("./data/classifiers/"+style+"_classifier.xml")
