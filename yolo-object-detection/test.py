import requests
import time
import picamera

txt1 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="a vehicle has been detected:"'
txt2 = 'https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendMessage?chat_id=-431037595&text="The number plate is: ~~~~~"'

with picamera.PiCamera() as input:
    input.resolution = (1024,768)
    input.start_preview()
    input.rotation = 180
    time.sleep(2)
    input.capture('foo.jpg')

requests.get(txt1)

files = {'photo':open('/home/pi/ANPR/yolo-object-detection/foo.jpg','rb')}

requests.post('https://api.telegram.org/bot1794107664:AAH8IZuCdcLDr_koMY24otS3K8WbAHs7Ljw/sendPhoto?chat_id=-431037595', files=files)

requests.get(txt2)

#print(resp.status_code)
