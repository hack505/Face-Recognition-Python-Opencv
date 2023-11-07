print("""
 _   _            _    ____   ___  ____  
| | | | __ _  ___| | _| ___| / _ \| ___| 
| |_| |/ _` |/ __| |/ /___ \| | | |___ \ 
|  _  | (_| | (__|   < ___) | |_| |___) |
|_| |_|\__,_|\___|_|\_\____/ \___/|____/ 
                                         
""")

# Import necessary libraries
import cv2
import numpy as np
import face_recognition
import os

# Define the directory where your images are stored
path = "faces"

# Create lists to store images, class names, and face encodings
images = []
classnames = []

# Get the list of image files in the directory
mylist = os.listdir(path)

# Print the list of image files for reference
print("List of image files:", mylist)

# Loop through the images and store them in the 'images' list, along with class names
for cl in mylist:
    curimg = cv2.imread(f"{path}/{cl}")
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

# Print the class names for reference
print("Class names:", classnames)

# Function to find face encodings for a list of images
def find_encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

# Get the known face encodings
encodelistknow = find_encodings(images)
print("Encoding completed...")

# Open the webcam (you can change the argument to '1' if '0' doesn't work)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Resize the captured frame for faster processing
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    facecurframe = face_recognition.face_locations(imgs)
    encodecurframe = face_recognition.face_encodings(imgs, facecurframe)

    for encodeface, faceloc in zip(encodecurframe, facecurframe):
        # Compare face encodings with known encodings
        macthes = face_recognition.compare_faces(encodelistknow, encodeface)
        facedis = face_recognition.face_distance(encodelistknow, encodeface)
        matchindex = np.argmin(facedis)

        if macthes[matchindex]:
            # Get the name of the recognized person
            name = classnames[matchindex].upper()
            y1, x2, y2, x1 = faceloc

            # Scale the coordinates for the original frame
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw a rectangle around the recognized face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Display the name of the recognized person
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with recognized faces
    cv2.imshow("Face Recognition", img)
    
    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


