import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2
import numpy as np

# load YOLOv8 model
model = YOLO('yolov8n.pt')
model.eval()

# register forward hook
features = {}
def hook(module, input, output):
    features[module._layer_name] = output

# mark layers with names and register hooks
layer_names = ['p1', 'p2', 'p3', 'p4', 'p5']
for i, m in enumerate(model.model.model):
    if m.__class__.__name__ == 'Detect':
        break  # stop at detection head
    m._layer_name = f'p{i+1}'
    m.register_forward_hook(hook)

# load image
img_path = 'carDemo.jpg'  # change to your image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# run forward
with torch.no_grad():
    model(img_tensor)

# save all layers
def save_feature_maps(fmap, name):
    fmap = fmap.squeeze(0)
    save_dir = f'features/{name}'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(fmap.shape[0]):
        ch = fmap[i].cpu()
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)
        plt.imshow(ch, cmap='inferno')
        plt.axis('off')
        plt.savefig(f'{save_dir}/ch{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

for name in features:
    save_feature_maps(features[name], name)
