# Real-time Face Recognition using AWS Lambda, Docker, and Raspberry Pi

## Setup and Execution

### S3:

* Create an S3 bucket (we used ‘cc-project-videos’ as the bucket name’) to upload every 0.5-second video from Raspberry Pi.

### DynamoDB:

* Create a DynamoDB table with ‘label’ as the partition key (we created a table named ‘student_table’).

* The items under the ‘label’ attribute should match the labels obtained from the machine learning model.

* Add other student information such as name, major, and year to the table.

### Docker image creation:

* Build the docker image using the command given below specifying the AWS parameters which will be used to access AWS services like S3 and DynamoDB.

    >docker build -t my_container --build-arg AWS_DEFAULT_REGION="us-east-1" --build-arg AWS_ACCESS_KEY_ID="your_id" --build-arg AWS_SECRET_ACCESS_KEY="your_key" .

* To test the docker image locally without deploying to lambda:
    * Run the docker image locally in a command prompt using the command:
        >docker run --rm -p 9000:8080 my_container
    * Then, invoke the lambda function locally by using http post requests targeted to the URL specified below, passing a test image as the body. A sample test method is given in http_post_local_testing.py in the project files.
        >http://localhost:9000/2015-03-31/functions/function/invocations

### Lambda:

* Upload the created Docker image to Amazon ECR.

* Create a Lambda function in AWS console using the docker image from the ECR repository.

* Increase the Memory Configuration of Lambda for faster processing.

### API Gateway:
* Create an API gateway(REST) and use that as a trigger for the lambda function. 

* Enable Lambda Proxy integration.

* Include 'image/jpeg' as Binary Media Type under the Settings tab.

### Raspberry Pi:
* Flash the Raspbian OS onto an SD card using a Raspberry Pi Imager and install it to the Raspberry device.

* Connect Raspberry Pi to a monitor using HDMI and use peripheral devices to boot up the device and connect it to a WiFi network (or use serial connection to do the same). 

* Set up an SSH server on the device and use a laptop to connect to the device from thereon.

* Copy the capture_video.py file to Raspberry Pi and execute the file by giving arguments for the video time interval and total duration in seconds (e.g., python capture_video.py 0.5 300)
