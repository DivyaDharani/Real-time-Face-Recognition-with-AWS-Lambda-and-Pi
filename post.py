import requests
import json
import base64

url="http://localhost:9000/2015-03-31/functions/function/invocations"    

with open('my_photo.jpeg', 'rb') as f:
     data = f.read()

data2 = base64.b64encode(data).decode("utf8")
data3=json.dumps({"body":data2,"other key":"hello"})    


res=requests.post(url, data=data3)
# res = requests.post(url,
#                     data=data,
#                     headers={'Content-Type': 'image/jpeg'})
                    
print(res.text)