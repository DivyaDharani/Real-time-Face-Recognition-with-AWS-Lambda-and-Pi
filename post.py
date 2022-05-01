import requests
with open('my_photo.jpeg', 'rb') as f:
    data = f.read()

url="https://8e13pzzye6.execute-api.us-east-1.amazonaws.com/prod"    

res = requests.post(url,
                    data=data,
                    headers={'Content-Type': 'image/jpeg'})
                    
print(res)