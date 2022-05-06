import torch
#import torchvision
import torchvision.transforms as transforms
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision.models as models
#from models.inception_resnet_v1 import InceptionResnetV1
#from urllib.request import urlopen
from PIL import Image
import json
import numpy as np
import argparse
import build_custom_model
import pickle
import os

def perform_image_recognition(img_path = "./data/test_me/val/angelina_jolie/1.png"):
     labels_dir = "./checkpoint/labels.json"
     model_path = "./checkpoint/model_vggface2_best.pth"

     # read labels
     with open(labels_dir) as f:
          labels = json.load(f)
     print(f"labels: {labels}")

     device = torch.device('cpu')

     if os.path.exists('model.pkl'):
          model = pickle.load(open('model.pkl', 'rb'))
     else:
          model = build_custom_model.build_model(len(labels)).to(device)
          pickle.dump(model, open('model.pkl', 'wb'))

     torch_loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
     model.load_state_dict(torch_loaded_dict['model'])
     model.eval()
     print(f"Best accuracy of the loaded model: {torch_loaded_dict['best_acc']}")

     img = Image.open(img_path)
     img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(device)
     outputs = model(img_tensor)
     _, predicted = torch.max(outputs.data, 1)
     result = labels[np.array(predicted.cpu())[0]]
     # print(predicted.data, result)

     img_name = img_path.split("/")[-1]
     img_and_result = f"({img_name}, {result})"
     print(f"Image and its recognition result is: {img_and_result}")
     return result

import time
if __name__ == "__main__":
     start_time = time.time()
     parser = argparse.ArgumentParser(description='Evaluate your customized face recognition model')
     parser.add_argument('--img_path', type=str, default="./data/test_me/val/angelina_jolie/1.png", help='the path of the dataset')
     args = parser.parse_args()
     img_path = args.img_path
     labels_dir = "./checkpoint/labels.json"
     model_path = "./checkpoint/model_vggface2_best.pth"


     # read labels
     with open(labels_dir) as f:
          labels = json.load(f)
     print(f"labels: {labels}")

     device = torch.device('cpu')

     if os.path.exists('model.pkl'):
          model = pickle.load(open('model.pkl', 'rb'))
     else:
          model = build_custom_model.build_model(len(labels)).to(device)
          pickle.dump(model, open('model.pkl', 'wb'))

     torch_loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
     model.load_state_dict(torch_loaded_dict['model'])
     model.eval()
     print(f"Best accuracy of the loaded model: {torch_loaded_dict['best_acc']}")


     img = Image.open(img_path)
     img_tensor = transforms.ToTensor()(img).unsqueeze_(0).to(device)
     outputs = model(img_tensor)
     _, predicted = torch.max(outputs.data, 1)
     result = labels[np.array(predicted.cpu())[0]]
     # print(predicted.data, result)


     img_name = img_path.split("/")[-1]
     img_and_result = f"({img_name}, {result})"
     print(f"Image and its recognition result is: {img_and_result}")
     print('Total time: ', time.time() - start_time, 'seconds')