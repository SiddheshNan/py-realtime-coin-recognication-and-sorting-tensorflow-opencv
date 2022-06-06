import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import serial
from datetime import datetime
import time
import mailgun


coin1 = 0
coin2 = 0
coin5 = 0
coin10 = 0
total = 0

count = 0

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
arduino = serial.Serial('/dev/ttyACM0', 9600)
print("loading...")

def detect(img):  # takes image path as arg
    global coin1, coin2,coin5, coin10, total
    image = Image.open(img)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    # image.show()
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    # print(prediction)
    prediction_new = prediction[0].tolist()
    detected_coin = prediction_new.index(max(prediction_new))
    detected_coin_acc = max(prediction_new)
    print("accuracy: " + str(detected_coin_acc))
    coin_is = ''
    if detected_coin == 0:
        coin_is = '2'
        coin2 +=1
        total +=2
    elif detected_coin == 1:
        coin_is = '5'
        coin5 += 1
        total += 5
    elif detected_coin == 2:
        coin_is = '10'
        coin10 += 1
        total += 10
    elif detected_coin == 3:
        coin_is = '1'
        coin1 += 1
        total += 1
    elif detected_coin == 4:
        coin_is = '1'
        coin1 += 1
        total += 1
    elif detected_coin == 5:
        coin_is = 'none'
    return coin_is


def doStart():
    global count
    arduino.write('X'.encode('ascii'))
    time.sleep(3)
    arduino.write('X'.encode('ascii'))
    ii = 0
    webcam = cv2.VideoCapture(2)
    if arduino.read().decode('ascii') == 'A':
        print("Image Acquired")
        cv2.namedWindow("Image")
        while True:
            check, frame = webcam.read()
            cv2.imshow("Image", frame)
            key = cv2.waitKey(10)
            ii += 1
            if ii >= 30 and key:
                img_name = "img_at_{}.jpg".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                cv2.imwrite(img_name, frame)
                #print("{} written!".format(img_name))
                count += 1
                print("Count : ", count)
                coin = detect(img_name)
                print('Detected coin is: ' + coin)
                print("Coins: Rs1: {} | Rs2: {} | Rs5: {} | Rs10: {} | Total: {}".format(coin1, coin2, coin5, coin10, total))
                print("------")
                if coin != 'none':
                    if coin == '10':
                        coin = '6'
                    arduino.write(coin.encode('ascii'))
                    time.sleep(1)
                    arduino.write('X'.encode('ascii'))
                else:
                    time.sleep(1)
                    arduino.write('X'.encode('ascii'))
                os.remove(img_name)
                ii = 0
            if cv2.waitKey(20) % 256 == 27:
                print("Esc Pressed.. Exiting..")
                break
            if cv2.waitKey(20) % 256 == ord('m'):
                print("sending mail.. Please wait ...")
                mailgun.mail("test@example.com", "coin detection" ,"Coins: Rs1: {} | Rs2: {} | Rs5: {} | Rs10: {} | Total: {}".format(coin1, coin2, coin5, coin10, total))
                print("mail sent...")

    webcam.release()
    cv2.destroyAllWindows()
    # delay proper
    # email on buton


if __name__ == '__main__':
    doStart()
