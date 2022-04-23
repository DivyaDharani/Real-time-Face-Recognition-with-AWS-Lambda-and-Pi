import sys
from eval_face_recognition import perform_image_recognition


def face_recognition_handler(event, context):
    name = 'Alice'
    major = 'Computer Science'
    year = '2021'
    print("Event received:", event)
    print("Context received:", context)
    return (name, major, year)

if __name__ == '__main__':
    print("The result is: ", perform_image_recognition())
    print(face_recognition_handler(None, None))