#Inspiration
#https://www.youtube.com/watch?v=3EBdT-0gvu8
#https://github.com/joeVenner/FaceRecognition-GUI-APP/blob/master/Detector.py

#Modified code below:
import cv2
import os

def start_capture(style):
        path = "./data/" + style
        num_of_images = 0
	#https://medium.com/@vipulgote4/guide-to-make-custom-haar-cascade-xml-file-for-object-detection-with-opencv-6932e22c3f0e
        detector = cv2.CascadeClassifier("./data/haarcascade_default.xml")
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        cap = cv2.VideoCapture(0)
        while True:

            ret, img = cap.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bag = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in bag:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                cv2.putText(img, "Bag Style Detected", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(img, str(str(num_of_images)+" images captured"), (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                new_img = img[y:y+h, x:x+w]
            cv2.imshow("BagDetection", img)
            key = cv2.waitKey(1) & 0xFF


            try :
                cv2.imwrite(str(path+"/"+str(num_of_images)+style+".jpg"), new_img)
                num_of_images += 1
            except :

                pass
            if key == ord("q") or key == 27 or num_of_images > 310:
                break
        cv2.destroyAllWindows()
        return num_of_images


#*********************************************************************************************************************************************************************************************

img = cv2.imread('RealBlue.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
kp, des = orb.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

cv2.imshow('ORB', kp_img)
cv2.waitKey()



#code:
