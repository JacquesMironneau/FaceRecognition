import cv2
import numpy as np
import face_recognition
import os

peoplePath = 'people'
encodingPath = 'encodings'
images = []


def detect_differences():
    """
    Find the missing encoded images by comparing the encoding and people folder
    returns the list of missing people
    """
    peopleName = [os.path.splitext(ppl)[0] for ppl in os.listdir(peoplePath)]
    npyName = [os.path.splitext(nypd)[0] for nypd in os.listdir(encodingPath)]

    return list(set(peopleName) - set(npyName))


def retrieve_encodings():
    """
    Retrieve encodings from the encodings folder
    this folder is used as the buffer to prevent the application to regenerate every 
    image encoding at start
    """
    myNumpyArrays, cNames = [], []
    npyFilesList = os.listdir(encodingPath)

    for file in npyFilesList:
        myNumpyArrays.append(np.load(rf"{encodingPath}/{file}"))
        cNames.append(os.path.splitext(file)[0])
        print(
            f'[{npyFilesList.index(file)+1}/{len(npyFilesList)}] Encoded file : {file[:-4]} loaded')

    return np.asarray(myNumpyArrays), cNames


def save_encodings(peopleNameList):
    """
    Save encoding as {name}.npy for every listed people 
    """
    imgs = [cv2.imread(rf'{peoplePath}/{people}.jpg')
            for people in peopleNameList]

    for img, people in zip(imgs, peopleNameList):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        file = rf'{encodingPath}/{people}.npy'
        np.save(file, np.asarray(encode))

        print(f'[{peopleNameList.index(people)+1}/{len(peopleNameList)}] Image : {people} encoded in {encodingPath}/{people}.npy')


if __name__ == "__main__":

    print("[1] Start encoding...")
    print("[2] Retrieving differences between images and encodings")

    save_encodings(detect_differences())
    encodeListKnown, classNames = retrieve_encodings()
    print("Encoding complete !")

    '''
    Live capture and face comparison
    '''

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        # Reduce img size for speeder treatment

        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        faceLocation = face_recognition.face_locations(imgSmall)
        encodesCurrentFrame = face_recognition.face_encodings(
            imgSmall, faceLocation)

        for encodeFace, faceLoc in zip(encodesCurrentFrame, faceLocation):
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDistance = face_recognition.face_distance(
                encodeListKnown, encodeFace)

            # print(faceDistance)
            matchIndex = np.argmin(faceDistance)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(f"La personne à l'écran est : {name}")
                y1, x2, y2, x1 = faceLoc
                # print(faceLoc)
                y1, x2, y2, x1 = 4*y1, 4*x2, 4*y2, 4*x1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 225, 255), 2)

        cv2.imshow("webcam", img)
        cv2.waitKey(1)
