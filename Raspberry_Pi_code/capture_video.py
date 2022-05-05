import picamera
import time
import cv2

VIDEO_DURATION = 60
NUMBER_OF_VIDEOS = 120

'''def capture_video():    
    with picamera.PiCamera() as camera:
        for filename in camera.record_sequence(
                'clip%03d.h264' % i for i in range(3)):
            print('Recording to %s' % filename)
            camera.wait_recording(0.5)'''

def capture_video():
    tic = time.perf_counter()
    print (f'initial tic = {tic}')
    initial_time = tic
    with picamera.PiCamera() as camera:
        for filename in camera.record_sequence(
                    'clip%03d.h264' % i for i in range(120)):
            print('Recording to %s' % filename)
            camera.wait_recording(0.5)
            tic = time.perf_counter()
            if tic - initial_time >= VIDEO_DURATION:
                print (f'final tic = {tic}')
                break

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

    if final_image is not None:
        image_name = video_name.removesuffix('h264') + 'jpg'
        cv2.imwrite(image_name, final_image) # save frame as JPEG file
        print(f'Save the final frame number:{count} and name: {image_name}')
    return final_image

def extract_frames(number_of_videos):
    for i in range(number_of_videos):
        video_name = 'clip%03d.h264' % i
        final_image = capture_frame(video_name)
        if final_image is None:
            break
    
if __name__ == "__main__":
    capture_video()
    extract_frames(NUMBER_OF_VIDEOS)