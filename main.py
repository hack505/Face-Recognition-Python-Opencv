"""
@author: Kamesh(hack505)
moulde required:
opencv-python
numpy
cmake
dlib
face_recognition
os
"""



import cv2
import numpy as np
import face_recognition
import os

path = "faces"
images =[]
classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

def find_encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknow = find_encodings(images)
#print(len(encodelistknow))
print("Encodeing completed..........")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs,facecurframe)

    for encodeface, faceloc in zip(encodecurframe,facecurframe):
        macthes = face_recognition.compare_faces(encodelistknow, encodeface)
        facedis = face_recognition.face_distance(encodelistknow, encodeface)
        #print(facedis)
        matchindex = np.argmin(facedis)
        

        if macthes[matchindex]:
            name = classnames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

            cv2.rectangle(img,(x1,y1),(x1,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2), (x2,y2), (0,255,0), cv2.FILLED)
            #cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            #cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,3, (255,255,255),2)
            cv2.putText(img, name, (x1,y2), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255),2)
            


    cv2.imshow("RECO", img)
    cv2.waitKey(1)


