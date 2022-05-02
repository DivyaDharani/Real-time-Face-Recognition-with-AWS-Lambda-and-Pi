import os
import json
import base64
from eval_face_recognition import perform_image_recognition

import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('student_table')

def face_recognition_handler(event, context):
    print('Face Recognition Handler!!')
    env_var = "AWS_LAMBDA_RUNTIME_API"
    if env_var in os.environ:
        print(f"{env_var} = {os.environ[env_var]}")
    else:
        print(f"Env var {env_var} not found")
    # name = 'Alice'
    # major = 'Computer Science'
    # year = '2021'
    # print("Event received:", event)
    # print("Context received:", context)
    file_content=event["body"]
    decode_content=base64.b64decode(file_content)
    # print(decode_content)
    
    with open('/tmp/hello.png', 'wb') as f:
        f.write(decode_content)

    a=perform_image_recognition('/tmp/hello.png')
    print(a)

    response = table.get_item(
        Key={
            'name': a
        }
    )
    result={}
    result['name']=a
    result['year']=response['Item']['year']
    result['major']=response['Item']['major']    

    #return result when testing locally.
    
    return {
        'body': json.dumps(result)
    }    