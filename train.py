import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
from torchvision import datasets,transforms,models
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest="data_dir",action="store",type=str,required=True)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.003)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=250)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    arguments = parser.parse_args()
    
    return arguments
    
    
def dataset_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader
    
    
def train_data_transformer(train_dir):
   train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data



def test_data_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def validation_data_transformer(valid_dir):
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    return validation_data

def check_for_gpu(gpu_argument):
    if gpu_argument=='cpu':
        return torch.device("cpu")
    elif gpu_argument=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("CUDA not found, using.")
        return device
    
    

def default_model(architecture="vgg19"):
    dictionary = {"vgg19" : models.vgg19(pretrained = True),
    "densenet121" : models.densenet121(pretrained = True),
    "resnet152" : models.resnet152(pretrained = True)}
    model_dict={"vgg19":25088,"densenet121":1024,"resnet152":2048}  
    model = dictionary[architecture]
    for param in model.parameters():
        param.requires_grad = False
    return model


def default_classifier(model, hidden_units,architecture):
    model_dict={"vgg19":25088,"densenet121":1024,"resnet152":2048}
    

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(model_dict[architecture], hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 150)),
                          ('relu2',nn.ReLU()),
                          ('dropout',nn.Dropout(0.5)),
                          ('fc3',nn.Linear(150,90)),
                          ('relu3',nn.ReLU()),
                          ('fc4',nn.Linear(90,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return classifier
    
    
def trainer(model, traindataloader, testdataloader, device, 
                  criterion, optimizer,epochs, print_every, steps):
    
    if type(epochs) == type(None):
        Epochs = 1
        print("Number of Epochs=1.")    
    running_loss=0
    print("Training process starting .............................\n")

    for epoch in range(epochs):
       for inputs, labels in traindataloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testdataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_probability, top_class = ps.topk(1, dim=1)
                    equals = (top_class == labels.view(*top_class.shape))
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testdataloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testdataloader):.3f}")
            running_loss = 0
            model.train()

    return model

    
    
    
def test_model(model,testdataloader,device):
    correct_preds = 0
    total_preds=0
    model.to(device)
    with torch.no_grad():
        for data in testdataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            _, predictions = torch.max(logps.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predictions == labels).sum().item()

    print("ACCURACY of the model is ",correct_preds*100/total_preds) 
    

            
def default_checkpoint(train_data,modelaftertraining,path='checkpoint.pth',structure ='vgg19', hidden_layer1=250,lr=0.003,epochs=1,dropout=0.5):
    
    modelaftertraining.class_to_idx = train_data.class_to_idx
    modelaftertraining.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':modelaftertraining.state_dict(),
                'class_to_idx':modelaftertraining.class_to_idx},
                path)

    
def main():
    print("this code downloads 3 pretrained model [vgg19,densenet121,resnet152] from which you can choose which model you want to train ")
    args = argparser()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_data_transformer(train_dir)
    validation_data = validation_data_transformer(valid_dir)
    test_data = test_data_transformer(test_dir)
    
    trainloader = dataset_loader(train_data)
    validationloader = dataset_loader(validation_data, train=False)
    testloader = dataset_loader(test_data, train=False)
    
    model = default_model(architecture=args.arch)
    
    model.classifier = default_classifier(model, hidden_units=args.hidden_units,architecture = args.arch)
    
    device = check_for_gpu(gpu_argument=args.gpu)
    model.to(device)
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.003
        print("Learning rate set as 0.003")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 3
    steps = 0
    
    modelaftertraining = trainer(model, trainloader, validationloader,device, criterion, optimizer, args.epochs, print_every, steps)
    print("....................................................................")
    print("Training completed")
    print("....................................................................")
    test_model(modelaftertraining, testloader, device)
   
    default_checkpoint(train_data,modelaftertraining,args.save_dir,args.arch,args.hidden_units,args.learning_rate,args.epochs,dropout = 0.5 )
    
    
if __name__ == '__main__': main()