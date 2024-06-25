#Clone Repository
!git clone https://github.com/misbah4064/face_recognition.git
%cd face_recognition

#Install Face Recognition from pip
!pip install face_recognition

#Passing images to train
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display


# face_1 = face_recognition.load_image_file("elon.jpg")
# face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_1 = face_recognition.load_image_file("john.png")
face_1_encoding = face_recognition.face_encodings(face_1)[0]

face_2 = face_recognition.load_image_file("ertugrul.jpeg")
face_2_encoding = face_recognition.face_encodings(face_2)[0]

known_face_encodings = [
    face_1_encoding,
    face_2_encoding
]
known_face_names = [
    "John",
    "Ertugrul"
]
print("Done learning and creating profiles")
