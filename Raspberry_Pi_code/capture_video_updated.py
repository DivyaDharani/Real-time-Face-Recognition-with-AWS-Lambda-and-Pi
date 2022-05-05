import picamera
import time
import cv2
import sys
import os

import boto3
from botocore.exceptions import ClientError
import requests


TOTAL_VIDEO_DURATION = 60 #in seconds
SINGLE_VIDEO_DURATION = 0.5
NUMBER_OF_VIDEOS = int(TOTAL_VIDEO_DURATION / SINGLE_VIDEO_DURATION)
VIDEO_PATH = 'Videos/clip%03d.h264'

#AWS parameters
URL = "https://5zkt8bff7d.execute-api.us-east-1.amazonaws.com/default/cc-project-lambda"
S3_BUCKET_NAME = 'cc-project-videos'

def capture_frame(video_name):
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    count = 0
    final_image = None
    while success:
        final_image = image
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    image_name = ''
    if final_image is not None:
        image_name = video_name.removesuffix('h264') + 'jpg'
        cv2.imwrite(image_name, final_image) # save frame as JPEG file
        #print(f'Saving the final frame number:{count} and name: {image_name}')
    return final_image, image_name

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


def get_face_recognition_result(image_filename):
    with open(image_filename, 'rb') as f:
        data = f.read()
    res = requests.post(URL,
                         data=data,
                         headers={'Content-Type': 'image/jpeg'})
    return res.text


def process_video(video_file_name, start_time):
    imageObj, image_file_name = capture_frame(video_file_name)
    if imageObj is not None and image_file_name != '':
        face_recog_result = get_face_recognition_result(image_file_name)
        latency = time.time() - start_time
        print(face_recog_result)
        print("Latency: {:.2f} seconds.".format(latency))
    #---- use multithreading here
    upload_file(video_file_name, S3_BUCKET_NAME)


def capture_and_process_video():
    #tic = time.perf_counter()
    #print (f'initial tic = {tic}')
    #initial_time = tic
    with picamera.PiCamera() as camera:
        for filename in camera.record_sequence(
                    VIDEO_PATH % i for i in range(NUMBER_OF_VIDEOS)):
            print('Recording to %s' % filename)
            start_time = time.time()
            camera.wait_recording(SINGLE_VIDEO_DURATION)
            #---- use multithreading here
            process_video(filename, start_time)  #PROCESSING
            # tic = time.perf_counter()
            # if tic - initial_time >= TOTAL_VIDEO_DURATION:
            #     print (f'final tic = {tic}')
            #     break


if __name__ == "__main__":
    n = len(sys.argv)
    if n >= 2:
        SINGLE_VIDEO_DURATION = float(sys.argv[1])
    if n >= 3:
        TOTAL_VIDEO_DURATION = float(sys.argv[2])

    #print('Single Video Duration:', SINGLE_VIDEO_DURATION, 'seconds')
    #print('Total Video Duration:', TOTAL_VIDEO_DURATION, 'seconds')

    capture_and_process_video()