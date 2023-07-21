import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets,transforms,models
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
import argparse
import PIL
from PIL import Image
import json


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',type=str,required=True)
    parser.add_argument('--checkpoint',type=str,required=True)
    parser.add_argument('--top_k',type=int,default = 5)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

    arguments = parser.parse_args()
    
    return arguments

def architecture(model_n,dropout,layer1,learnrate):
    dictionary = {"vgg19" : models.vgg19(pretrained = True),
    "densenet121" : models.densenet121(pretrained = True),
    "resnet152" : models.resnet152(pretrained = True)}
    model_dict={"vgg19":25088,"densenet121":1024,"resnet152":2048}
    model = dictionary[model_n]
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model_dict[model_n], layer1)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(layer1, 150)),
                          ('relu2',nn.ReLU()),
                          ('dropout',nn.Dropout(dropout)),
                          ('fc3',nn.Linear(150,90)),
                          ('relu3',nn.ReLU()),
                          ('fc4',nn.Linear(90,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)
    
    return model,criterion,optimizer
    
    
def check_for_gpu(gpu_argument):
    if gpu_argument=='cpu':
        return torch.device("cpu")
    elif gpu_argument=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA not found, using.")
        return device
    
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = architecture(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pilimage = Image.open(image)
    transformings = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    imagetensor = transformings(pilimage)
    
    return imagetensor



def predict(image_path, model,device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    #model.eval()
    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    processed_image.to(device)
    
    with torch.no_grad():
        logps = model.forward(processed_image.to(device))
        ps = torch.exp(logps)
    
    return ps.topk(topk, dim=1)




def main():
    
    args = argparser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    
    device = check_for_gpu(gpu_argument=args.gpu);
    
    probability,indexes =  predict(args.image, model,device,args.top_k)
    
    print(probability[0])
    flower_name = [cat_to_name[str(index+1)] for index in np.array(indexes[0])]
    print(flower_name)

if __name__ == '__main__': main()