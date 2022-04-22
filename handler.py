import sys
import os

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
    os.system(f"python face_recognition_modules/eval_face_recognition.py")