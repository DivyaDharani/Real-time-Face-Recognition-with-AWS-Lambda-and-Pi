import os
from eval_face_recognition import perform_image_recognition

def face_recognition_handler(event, context):
    env_var = "AWS_LAMBDA_RUNTIME_API"
    if env_var in os.environ:
        print(f"{env_var} = {os.environ[env_var]}")
    else:
        print(f"Env var {env_var} not found")
    name = 'Alice'
    major = 'Computer Science'
    year = '2021'
    print("Event received:", event)
    print("Context received:", context)
    return (name, major, year)

if __name__ == '__main__':
    print("The result is: ", perform_image_recognition())
    print(face_recognition_handler(None, None))