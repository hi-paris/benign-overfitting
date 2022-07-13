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
from scipy.io.arff import loadarff

parser = argparse.ArgumentParser(description='')
# 1279
# "cuda:0" if torch.cuda.is_available() else
parser.add_argument('--input_size', dest='input_size', type=int,default=11, help='')
parser.add_argument('--output_size', dest='output_size', type=int, default=1, help='')

parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='# of epoch')

parser.add_argument('--start_size', dest='start_size', type=int, default=1, help='')
parser.add_argument('--max_size', dest='max_size', type=int, default=1600, help='maximum incremental size')
parser.add_argument('--noise_percentage', dest='noise_percentage', type=int, default=0, help='')
parser.add_argument('--nb_trials', dest='nb_trials', type=int, default=4, help='')

parser.add_argument('--device', dest='device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), help='device')

parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='lr')

parser.add_argument('--model_name', dest='model_name', type=str, default="nn", help='')
parser.add_argument('--optim_name', dest='optim_name', type=str, default="adam", help='')
parser.add_argument('--loss_name', dest='loss_name', type=str, default="mse", help='')

parser.add_argument('--dataset_name', dest='dataset_name', type=str, default="BNG_wine_quality", help='')

parser.add_argument('--save_dir', dest='save_dir', type=str, default="/content/drive/MyDrive/bening-overfitting/saved_results", help='device')

parser.add_argument('--json_name', dest='json_name', type=str, default="results.json", help='')

args = parser.parse_args()

destination_folder=args.save_dir+"/"+args.dataset_name+"_"+args.model_name+"_"+str(args.epochs)+"_"+str(args.max_size)+"_"+str(args.nb_trials)+"_"+args.loss_name+"_"+args.optim_name+"_"+str(args.lr)+"_"+str(args.noise_percentage)
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

  # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
  # column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
  #                 'Acceleration', 'Model Year', 'Origin']

  # raw_dataset = pd.read_csv(url, names=column_names,
  #                           na_values='?', comment='\t',
  #                           sep=' ', skipinitialspace=True) 

  data = loadarff('C:/Users/ykemiche/OneDrive - Capgemini/Desktop/Hi_Paris/benign-overfitting/data/BNG_wine_quality.arff')
  raw_dataset = pd.DataFrame(data[0])

  dataset = raw_dataset.copy()
  dataset = dataset.dropna()
  # dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
  # dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
  dataset.iloc[:,:11]=(dataset.iloc[:,:11]-dataset.iloc[:,:11].mean())/dataset.iloc[:,:11].std()

  train_features = dataset.copy()
  train_labels = train_features.pop('quality')

  X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels)

  # Add label noise
  noisy_elements=int(len(y_train)*args.noise_percentage/100)
  noise=np.random.normal(0,1,noisy_elements)
  y=np.concatenate([noise, np.zeros(len(y_train)-len(noise))])
  np.random.shuffle(y)
  y_train=y_train+y


  X_train = torch.Tensor(X_train.values.astype(np.float32))
  X_train=X_train.to(args.device)

  y_train = torch.Tensor(y_train.values.astype(np.float32)).view(y_train.shape)
  y_train=y_train.to(args.device)

  X_test = torch.Tensor(X_test.values.astype(np.float32))
  X_test=X_test.to(args.device)

  y_test = torch.Tensor(y_test.values.astype(np.float32)).view(y_test.shape)
  y_test=y_test.to(args.device)

  ############################ TRAINING ##########################################

  saved_values={}
  keys_errors = ["Train_Errors","Test_Errors"]
  keys_trials=[i for  i in range(args.nb_trials)]
  saved_values={key: {key: np.zeros((args.max_size-args.start_size,args.epochs)) for key in keys_errors} for key in keys_trials}

  for key in saved_values:
    saved_values[key]["Test_kernel_Errors"]=np.zeros((args.max_size-args.start_size))
    saved_values[key]["Train_kernel_Errors"]=np.zeros((args.max_size-args.start_size))


  for k in tqdm(range(args.start_size,args.max_size)):
      # print("********** TRAINING WIDTH: ",k)

      for trial in range(args.nb_trials):
        model=NN(args.input_size,args.output_size,k)
        model.to(args.device)
        fnet, params ,buffers =make_functional_with_buffers(model)

        # loss
        loss_fn = torch.nn.MSELoss() 
        # optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
        # optimizer
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        training_from_df(X_train,y_train,X_test,y_test,trial,args.epochs,model,loss_fn,optim,k-args.start_size,params,buffers,fnet,saved_values,json_file)

        # try:
        #   training_from_df(X_train,y_train,X_test,y_test,trial,args.epochs,model,loss_fn,optim,k-args.start_size,params,buffers,fnet,saved_values,json_file)

        # except LinAlgError as e:
        #         pass
