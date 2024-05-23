import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from src.depth.depth_anything.dpt import DepthAnything
from src.depth.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class Depth():
    def __init__(self, device = "cuda"):
        self.device = device
        self.load_model()
        self.get_transform()
    
    def load_model(self):
        self.model = DepthAnything({'encoder' : "vits", 'features': 64, 'out_channels' : [48, 96, 192, 384]})
        self.model.load_state_dict(torch.load("/home/tinozg/Documents/Ivy/model/depth/vits.pth"))
        self.model = self.model.to(self.device)

    def get_transform(self):
        self.transform = Compose([Resize(width = 518, height = 518, resize_target = False, keep_aspect_ratio = True,
                                         ensure_multiple_of = 14, resize_method = 'lower_bound', image_interpolation_method = cv2.INTER_CUBIC,),                                   
                                  NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  PrepareForNet(),])
                            
    def get_depth(self, img):
        img = cv2.resize(img, (640, 480))

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth = self.model(image)
        
            depth = F.interpolate(depth[None], (h, w), mode = 'bilinear', align_corners = False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # print(depth.max(), depth.min())
            return depth_color

if __name__ == "__main__":
    depth = Depth()
    cap = cv2.VideoCapture(2)

    while cap.isOpened():
        ret, raw_image = cap.read()

        if not ret:
            break

        res = depth.get_depth(raw_image)
        # cv2.imshow("out", res)
        cv2.imwrite("out.jpg", res)

        # Press q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()