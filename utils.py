
import torch
from tqdm import tqdm
import json
import numpy as np
from ntk.ntk import *
from functorch import make_functional,make_functional_with_buffers, vmap, vjp, jvp, jacrev


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# def fnet_single(params, x):
    
#     return fnet(params, x.unsqueeze(0)).squeeze(0)
# def empirical_ntk(fnet_single, params, x1, x2):
#     # Compute J(x1)
#     jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
#     jac1 = [j.flatten(2) for j in jac1]
    
#     # Compute J(x2)
#     jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
#     jac2 = [j.flatten(2) for j in jac2]
    
#     # Compute J(x1) @ J(x2).T
#     result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
#     result = result.sum(0)
#     return result



def training_from_df(X_train,y_train,X_test,y_test,trial,epochs,model,loss_func,optimizer,k,params,buffers,fnet,saved_values,json_path):


  scores = []

  with open(json_path, 'w') as f:
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):

        s_predicted = model.forward(X_train)
        loss = loss_func(s_predicted.reshape(y_train.shape[0]), y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        saved_values[trial]["Train_Errors"][k-1,epoch]=loss.item()

        # score = r2_score(s_predicted.detach().numpy(), s)
        # scores.append(score)

        #Validation
        with torch.no_grad():
          s_val_predicted = model.forward(X_test)
          val_loss = loss_func(s_val_predicted.reshape(y_test.shape[0]), y_test)
          saved_values[trial]["Test_Errors"][k-1,epoch]=val_loss.item()
    #  Test     
    try:
      test_kernel_predicted=compute_kxkny(X_train,X_test,y_train,params,buffers,fnet)
      test_kernel_loss=loss_func(test_kernel_predicted.reshape(y_test.shape[0]), y_test)
      saved_values[trial]["Test_kernel_Errors"][k-1]=test_kernel_loss.item()
    
    except:
      saved_values[trial]["Test_kernel_Errors"][k-1]=1


    # Train
    try:
      train_kernel_predicted=compute_kxkny(X_train,X_train,y_train,params,buffers,fnet)
      train_kernel_loss=loss_func(train_kernel_predicted.reshape(y_train.shape[0]), y_train)
      saved_values[trial]["Train_kernel_Errors"][k-1]=train_kernel_loss.item()
    
    except:
      saved_values[trial]["Train_kernel_Errors"][k-1]=1


    json.dump(saved_values, f, indent=4,cls=NumpyEncoder)

def train_from_loader(epoch,model,loss_fn,optim,train_loader,device):
    # model.train()
    loss_epoch = 0
    true_pre_epoch = 0
    correct = 0

    for i,(img,label) in enumerate(train_loader):

        img,label = img.to(device),label.to(device)
        output = model.forward(img)

        loss = loss_fn(output,label)
        loss.backward()

        optim.step()
        optim.zero_grad()
        loss_epoch += loss.data

        pre = torch.argmax(output, dim=1)
        num_true = (pre == label).sum()
        true_pre_epoch += num_true
        correct += label.shape[0]

        # if (i+1)%100 == 0:
        #     print('epoch {} iter {} loss : {}'.format(epoch,i+1,loss_epoch/(i+1)))

        # if (i+1)%200 == 0:
        #     acc = true_pre_epoch/correct
        #     print('epoch {} iter {} train_acc : {}'.format(epoch,i+1,acc))
            
    optim.param_groups[0]['lr'] = optim.param_groups[0]['lr']/2
    return loss_epoch/(i+1)


def test(model,test_loader,loss_fn,device):
# test
    model.eval()
    num = 0
    labels = 0
    loss_test=0

    for j,(img,label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        output = model(img)
        num += (torch.argmax(output,dim=1).data == label.data).sum()
        labels += label.shape[0]
        loss = loss_fn(output,label)
        loss_test += loss.data

    # print('test loss : {}'.format(loss_test/(j+1)))
    # print('test_acc : {} '.format(num/labels))
    return loss_test/(j+1)
