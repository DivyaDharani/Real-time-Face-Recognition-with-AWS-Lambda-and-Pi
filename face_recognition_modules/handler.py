import sys
import os
from face_recognition_modules.eval_face_recognition import perform_image_recognition


def face_recognition_handler(event, context):
    name = 'Alice'
    major = 'Computer Science'
    year = '2021'
    print(name, major, year)
    print("Event received:", event)
    print("Context received:", context)
    return (name, major, year)

if __name__ == '__main__':
    print(sys.argv)
    perform_image_recognition()
    os.system(f"python eval_face_recognition.py")