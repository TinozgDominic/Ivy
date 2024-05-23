import paho.mqtt.client as mqtt
import base64
import numpy as np
import cv2
import time
from PIL import Image

from queue import Queue
from src.faster_whisper.faster_whisper import FasterWhisper
from src.stable_diffusion.stable_diffusion import stable_diffusion2_1
from src.utility.utils import encode_image

from multiprocessing import Process

class MQTT():
    def __init__(self, whisper: FasterWhisper):
        self.whisper = FasterWhisper()
        self.running = True

        self.broker = "test.mosquitto.org"
        self.camera_topic      = "ivy/cam"
        self.voice_topic       = "ivy/mic"
        self.draw_input_topic  = "ivy/draw_input"
        self.draw_output_topic = "ivy/draw_output"
        self.empty_topic       = "ivy/empty"
        
        self.client_cam = mqtt.Client()
        self.client_cam.on_connect = self.on_connect_cam
        self.client_cam.on_message = self.on_message_cam
        self.client_cam.connect(self.broker, 1883)
        self.client_cam.loop_start()

        self.client_mic = mqtt.Client()
        self.client_mic.on_connect = self.on_connect_mic
        self.client_mic.on_message = self.on_message_mic
        self.client_mic.connect(self.broker, 1883)
        self.client_mic.loop_start()

        self.client_draw_input = mqtt.Client()
        self.client_draw_input.on_connect = self.on_connect_draw_input
        self.client_draw_input.on_message = self.on_message_draw_input
        self.client_draw_input.connect(self.broker, 1883)
        self.client_draw_input.loop_start()

        self.data_queue = Queue()

        self.client_publish = mqtt.Client()
        self.client_publish.on_connect = self.on_connect_publish
        self.client_publish.connect(self.broker, 1883)
        self.client_publish.loop_start()

        self.frame = np.zeros((640, 640, 3), np.uint8)

        self.maintain_process = Process(target = self.maintain)
        self.maintain_process.start()

    def on_connect_cam(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        self.client_cam.subscribe(self.camera_topic,qos =0)

    def on_connect_mic(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        self.client_mic.subscribe(self.voice_topic,qos =0)
        
    def on_connect_draw_input(self, client, userdata, flags, rc):
        print(f"Connected with result code {rc}")
        self.client_draw_input.subscribe(self.draw_input_topic, qos = 0) 

    def on_connect_publish(self, client_publish, userdata, flags, rc):
        print(f"Connected with result code {rc}")

    def on_message_cam(self, client, userdata, msg):
        img = base64.b64decode(msg.payload)
        npimg = np.frombuffer(img, dtype=np.uint8)
        self.frame = cv2.imdecode(npimg, 1)
        self.frame = cv2.resize(self.frame, (640, 640))

    def on_message_mic(self, client, userdata, msg):
        data = msg.payload
        self.data_queue.put(data)
    
    def on_message_draw_input(self, client, userdata, msg):
        data = msg.payload
        text = self.whisper.transcribe(data)
        print(text)

        # img = stable_diffusion2_1(text)

        # img = img.resize((320, 240), Image.BICUBIC)
        # img = np.array(img)

        # encoded_image = encode_image(img)

        # self.upload_image(encoded_image)
        
    def upload_image(self, encoded_image):
        self.client_publish.publish(self.draw_output_topic, encoded_image, retain = False, qos = 0)

    def maintain(self):
        while self.running:
            self.client_publish.publish(self.empty_topic, "", retain = False, qos = 0)
            time.sleep(30)
        print("Stop")
    
    def disconnect(self):
        self.client_cam.loop_stop()
        self.client_mic.loop_stop()

        self.client_cam.disconnect()
        self.client_mic.disconnect()
        print("Disconnected successfully!")



   