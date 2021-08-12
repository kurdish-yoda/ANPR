import requests


txt1 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="a vehicle has been detected:"'
txt2 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="The number plate is: ~~~~~"'
files={'photo':open('/home/pi/ANPR/yolo-object-detection/test.jpg', 'rb')}

requests.get(txt1)

requests.post('https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendPhoto?chat_id=-431037595', files=files)

requests.get(txt2)

#print(resp.status_code)
