import torch
import torch.nn as nn
import requests
import os

def download_file(url, folder_path, file_name):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Full path where the file will be saved
    file_path = os.path.join(folder_path, file_name)
    
    # Stream the download to save memory
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive new chunks
                file.write(chunk)
    
    print(f"File downloaded successfully and saved to {file_path}")



def Dino_Model(**kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14',pretrained=False)
    model_path = 'stratification/classification/models'
    url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
    file_name = 'dinov2_vitl14_pretrain.pth'
    download_file(url, model_path, file_name)
    model.load_state_dict(torch.load(os.path.join(model_path,file_name)))
    model.activation_layer_name = 'head'
    return model
