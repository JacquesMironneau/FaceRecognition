import cv2
import numpy as np
import face_recognition


imgMarie = face_recognition.load_image_file(r'data/marie.jpg')
imgMarie = cv2.cvtColor(imgMarie, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file(r'data/marie2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


imgMe = face_recognition.load_image_file(r'data/me.jpg')
imgMe = cv2.cvtColor(imgMe, cv2.COLOR_BGR2RGB)

# position de la tête de Marie
faceLocation = face_recognition.face_locations(imgMarie)[0]
# encode la tête sur 128 valeurs x)
encodeMarie = face_recognition.face_encodings(imgMarie)[0]

cv2.rectangle(imgMarie, (faceLocation[3], faceLocation[0]),
              (faceLocation[1], faceLocation[2]), (255, 15, 255), 2)


faceLocationTest = face_recognition.face_locations(imgTest)[0]
# encode la tête sur 128 valeurs x)
encodeMarieTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]),
              (faceLocationTest[1], faceLocationTest[2]), (255, 15, 255), 2)

faceLocationMe = face_recognition.face_locations(imgMe)[0]
# encode la tête sur 128 valeurs x)
encodeMe = face_recognition.face_encodings(imgMe)[0]

cv2.rectangle(imgMe, (faceLocation[3], faceLocation[0]),
              (faceLocation[1], faceLocation[2]), (255, 15, 255), 2)


result = face_recognition.compare_faces(
    [encodeMarie, encodeMe], encodeMarieTest)
faceDistance = face_recognition.face_distance(
    [encodeMarie, encodeMe], encodeMarieTest)
print(faceDistance)
print(result)

cv2.putText(imgTest, f'{result}, {round(faceDistance[0],2)}',
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Marie Vialle', imgMarie)
cv2.imshow('Marie Vialle tested ', imgTest)
cv2.imshow('Me', imgMe)
cv2.waitKey(0)
