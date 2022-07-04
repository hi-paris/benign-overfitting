from ast import arg
import torch
import os
import time
from models.neural_network import *
import numpy as np
from torch.utils.data import Dataset,DataLoader
import math
import torchvision
import argparse
from glob import glob
import pandas as pd
from tqdm import tqdm
from utils import *
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.resnet import make_resnet18k

parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else
# parser.add_argument('--input_size', dest='input_size', type=int,default=9, help='')
parser.add_argument('--output_size', dest='output_size', type=int, default=100, help='')

parser.add_argument('--epochs', dest='epochs', type=int, default=4000, help='# of epoch')

parser.add_argument('--start_size', dest='start_size', type=int, default=1, help='')
parser.add_argument('--max_size', dest='max_size', type=int, default=64, help='maximum incremental size')
parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device')

parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="resnet", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="adam", help='')

parser.add_argument('--dataset_name', dest='dataset_name', type=str, default="cifar100", help='')
parser.add_argument('--save_dir', dest='save_dir', type=str, default="/content/drive/MyDrive/bening-overfitting/saved_results", help='device')
parser.add_argument('--json_name', dest='json_name', type=str, default="results.json", help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.dataset_name+"_"+args.model_name+"_"+str(args.epochs)+"_"+str(args.max_size)+"_"+args.optim_name+"_"+str(args.lr)
destination_folder_plots=destination_folder+"/plots"

dest = os.path.exists(destination_folder)
plots=os.path.exists(destination_folder_plots)

if not dest:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder)

if not plots:
  # Create a new directory because it does not exist
  os.makedirs(destination_folder_plots)

json_file=destination_folder+"/"+args.json_name


if __name__ == '__main__':

  ############################ DATA ##########################################
  
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32,padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    # dataset
    train_dataset = torchvision.datasets.CIFAR100(
    root = 'data',
    train = True,
    transform = train_transform,
    download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
    root = 'data',
    train = False,
    transform =test_transform,
    download=True
    )

    # dataloader
    train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
    )

    test_loader = DataLoader(
    test_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=2
    )

    train_loss_values_wrt_size=[]
    test_loss_values_wrt_size=[]

    saved_values={}
    saved_values["Train_Errors"]=np.zeros((args.max_size-args.start_size,args.epochs))
    saved_values["Test_Errors"]=np.zeros((args.max_size-args.start_size,args.epochs))

    with open(json_file, 'w') as f:
        for k in range(args.start_size,args.max_size):
            print("********** TRAINING WIDTH: ",k)
            for epoch in tqdm(range(args.epochs)):
                
                model=make_resnet18k(k,args.output_size)
                model.to(args.device)
                # fnet, params ,buffers =make_functional_with_buffers(model)

                # loss
                loss_fn = torch.nn.CrossEntropyLoss()    

                # optimizer
                optim = torch.optim.Adam(model.parameters(), lr=args.lr)

                # train,val,kernel=training_from_df(X_train,y_train,X_test,y_test,args.epochs,model,loss_fn,optim,k-args.start_size,params,buffers,fnet,saved_values,json_file)
                train_loss=train_from_loader(model,loss_fn,optim,train_loader,args.device)
                val_loss=test(model,test_loader,loss_fn,args.device)

                saved_values["Train_Errors"][k-1,epoch]=train_loss.item()
                saved_values["Test_Errors"][k-1,epoch]=val_loss.item()

                # train_loss_values_wrt_size.append(train_loss)
                # test_loss_values_wrt_size.append(val_loss)

            # kernel_loss_values_wrt_size.append(kernel)

            # for epoch in tqdm(range(args.epoch)):C
            #     # print('---epoch{}---'.format(epoch))

            #     train_loss=train(epoch,model,loss_fn,optim)
            #     test_loss=test(model)
    json.dump(saved_values, f, indent=4,cls=NumpyEncoder)


    width_list=[width for width in range(args.start_size,args.max_size)]
    train_errs = np.array([M[-1] for M in saved_values['Train_Errors']])
    test_errs = np.array([M[-1] for M in saved_values['Test_Errors']])

    fig, ax = plt.subplots()
    ax.plot(width_list, test_errs, label='Test Error')
    ax.plot(width_list, train_errs, label='Train Error')
    # ax.plot(width_list, kernel_loss_values_wrt_size, label='NTK linear approx')
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Test/Train Error")
    ax.set_title("Learning curves")
    ax.legend()
    fig.savefig(destination_folder_plots+'/learning_curves.png')   
    plt.close(fig) 

