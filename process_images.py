import os
import cv2


def process_directory(fromdir, todir):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    files = os.listdir(fromdir)
    files = [f for f in files if '.jpg' in f]
    for file in files:
        image = cv2.imread(os.path.join(fromdir, file))
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        image = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(todir, file), image)


if __name__ == '__main__':
    process_directory('data/makeup', 'data/makeup_processed')
