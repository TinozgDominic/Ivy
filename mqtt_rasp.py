import cv2
import paho.mqtt.client as mqtt
import base64
import time

class MQTT():
    def __init__(self):
        self.broker = "test.mosquitto.org"
        self.topic  = "ivy/cam_pi"

        self.camera = cv2.VideoCapture(0)

        self.client = mqtt.Client()
        self.client.connect(self.broker, 1883)
        self.client.loop_start()

        self.sending_image = True

    def send_image(self, im_size = 320, quality = 20):
        # Read the frame
        _, frame = self.camera.read()
        if not _:
            return
        frame = cv2.resize(frame, (im_size, im_size))

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        encoded = base64.b64encode(buffer)

        # Publish
        self.client.publish(self.topic, encoded, retain = False, qos = 0)

        time.sleep(0.08)
    
    def disconnect(self):
        self.camera.release()
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected successfully!")

if __name__ == "__main__":
    mq = MQTT()
    st = time.perf_counter()
    while True:
        mq.send_image()
        # if time.perf_counter() - st >= 10:
        #     mq.disconnect()
        #     break