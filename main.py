from src.mqtt.mqtt import MQTT
from src.yolov8.yolov8 import Yolov8
from src.depth.depth import Depth
from src.faster_whisper.faster_whisper import FasterWhisper
# from src.i2t.i2t import I2T
import cv2


if __name__ == "__main__":
    # ylv8 = Yolov8(model = "m")
    # depth = Depth()
    whisper = FasterWhisper()
    mqtt = MQTT(whisper = whisper)

    while True: 
        try:
            continue
            # img = mqtt.frame
            # dep = depth.get_depth(img)
            # predict, img = ylv8.detect(img)   
            
            # cv2.imshow("res", img)
            # cv2.imshow("dep", dep)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     mqtt.disconnect()
            #     break
            # cv2.imwrite("out.jpg", img)
            # cv2.imwrite("dep.jpg", dep)
        except KeyboardInterrupt:
            mqtt.disconnect()
            break