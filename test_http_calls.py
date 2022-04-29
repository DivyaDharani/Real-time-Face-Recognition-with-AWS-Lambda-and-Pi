import requests
import json

url = "http://localhost:9000/2015-03-31/functions/function/invocations"
myobj = {'somekey': 'somevalue'}
data = json.dumps(myobj)

x = requests.post(url, data = data)

print(x.text)