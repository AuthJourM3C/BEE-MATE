import requests

#print('attempting to send audio')
url = 'http://m3capps.jour.auth.gr/audioapi'
with open('rms.wav', 'rb') as file:
    data = {'uuid':'-jx-1', 'alarmType':1, 'timeDuration':10}
    files = {'messageFile': file}
    req = requests.post(url, files=files, json=data)
    print(req.status_code)
    print(req.text)