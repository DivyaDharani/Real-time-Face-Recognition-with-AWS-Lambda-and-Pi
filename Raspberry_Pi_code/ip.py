import socket
import boto3
import os
import subprocess as sp

ip_address = sp.getoutput('hostname -I')
print(ip_address)

s3 = boto3.client('s3')
txt_data = ip_address.encode()
s3.put_object(Bucket='my-test-bucket-dd',Key='IP Address',Body=txt_data)