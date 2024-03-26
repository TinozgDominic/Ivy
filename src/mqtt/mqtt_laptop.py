import base64
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time

class MQTT():
    def __init__(self):
        self.broker = "test.mosquitto.org"
        self.topic  = "ivy/cam_pi"

        self.client = mqtt.Client()

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(self.broker, 1883)
        self.client.loop_start()
        self.image = np.zeros((640, 640, 3), dtype = np.uint8)

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        client.subscribe(self.topic, qos = 0)

    def on_message(self, client, userdata, msg):
        npimg = np.frombuffer(base64.b64decode(msg.payload), dtype=np.uint8)
        frame = cv2.imdecode(npimg, 1)
        self.image = cv2.resize(frame, (640, 640))
        # cv2.imwrite("strem.jpg", self.image)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected successfully!")
        

if __name__ == "__main__":
    mq = MQTT()

    import time

    st = time.perf_counter()

    while True:
        if time.perf_counter() - st >= 10:
            mq.disconnect()
            break
