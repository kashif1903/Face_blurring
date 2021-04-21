from face_blurring import anonymize_face_pixelate
from face_blurring import anonymize_face_simple
import numpy as np
import cv2
import argparse
ap=argparse.ArgumentParser()
# ap.add_argument("-i","--image",required=True,help="path to input image")
ap.add_argument("-m","--method",type=str,default="simple",choices=["simple","pixelated"],help="face blurring/anonymizing method")
ap.add_argument("-b", "--blocks", type=int, default=20,help="# of blocks for the pixelated blurring method")
args=vars(ap.parse_args())
print("[INFO] loading face detector model...")
face_cascade=cv2.CascadeClassifier("C:/Users/Lenovo/PycharmProjects/OpenCVPython/Resources/haarcascade_frontalface_default.xml")
cap= cv2.VideoCapture(0)
cap.set(3,720) # for width id=3
cap.set(4,480) # for height id=4
cap.set(10,50) # for brigthness id=10
while True:
    success,image=cap.read()
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.05, 5)
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        # 		# check to see if we are applying the "simple" face blurring
        # 		# method
        if args["method"] == "simple":
            image[y:y + h, x:x + w] = anonymize_face_simple(face, factor=3.0)
        # 		# otherwise, we must be applying the "pixelated" face
        # 		# anonymization method
        else:
            image[y:y + h, x:x + w] = anonymize_face_pixelate(face, blocks=args["blocks"])
            # 		# store the blurred face in the output image
            image[y:y + h, x:x + w] = face
    cv2.imshow("Output",image)
    if cv2.waitKey(16) & 0xFF==ord('q'):
        break