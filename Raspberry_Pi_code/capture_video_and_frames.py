import picamera
import time
import boto3
from botocore.exceptions import ClientError
import os
import requests
import threading

VIDEO_DURATION = 20
NUMBER_OF_SCREENSHOTS = 40
BUCKET_NAME = 'bucket_name'
URL="https://8e13pzzye6.execute-api.us-east-1.amazonaws.com/prod" 
VIDEO_FILENAME = 'full_video.h264'

def capture_video_and_frames():
    tic = time.perf_counter()
    print (f'initial tic = {tic}')
    initial_time = tic
    with picamera.PiCamera() as camera:
        camera.start_preview()
        #video_filename = 'clip%03d.h264' % i
        camera.start_recording(VIDEO_FILENAME)
        print('Recording to %s' % VIDEO_FILENAME)
        for i in range(NUMBER_OF_SCREENSHOTS):
            image_filename = 'clip%03d.jpg' % i
            camera.wait_recording(0.5)
            camera.capture(image_filename, use_video_port=True)
            #camera.wait_recording(0.25)
            #camera.stop_recording()
            tic = time.perf_counter()
            #download_thread = threading.Thread(target=get_face_recognition_result, name="get_face_recognition_result", args=image_filename)
            #download_thread.start()
            if tic - initial_time >= VIDEO_DURATION:
                print (f'final tic = {tic} and count = {i}')
                camera.stop_recording()
                upload_file(VIDEO_FILENAME, BUCKET_NAME)
                return i
    camera.stop_recording()
    print (f'final tic = {tic} and count = {NUMBER_OF_SCREENSHOTS}')
    upload_file(VIDEO_FILENAME, BUCKET_NAME)
    return NUMBER_OF_SCREENSHOTS


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        print(f'Uploaded file {file_name}')
    except ClientError as e:
        print(f'Exception occurred while uploading {file_name}', e)
        return False
    return True

def upload_all_videos(count):
    for i in range(count):
        video_filename = 'clip%03d.h264' % i
        upload_file(video_filename, BUCKET_NAME)
        
def get_face_recognition_result(image_filename):
    
    with open(image_filename, 'rb') as f:
        data = f.read()   
    res = requests.post(URL,
                        data=data,
                        headers={'Content-Type': 'image/jpeg'})
    print(res)
            
if __name__ == "__main__":
    count = capture_video_and_frames()
    #upload_all_videos(count)
    