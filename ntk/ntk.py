from functorch import make_functional,make_functional_with_buffers, vmap, vjp, jvp, jacrev
import numpy as np
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fnet_single(params,buffers,x,fnet):
    return fnet(params,buffers, x.unsqueeze(0)).squeeze(0)


def compute_jacobian(x,params,buffers,fnet):
  
  jac1 = vmap(jacrev(fnet_single), (None,None, 0,None))(params,buffers,x,fnet)
  jac1 = [j.flatten(2) for j in jac1]

  return jac1


def compute_kernel(x1,x2,params,buffers,fnet):

  jac1 = compute_jacobian(x1,params,buffers,fnet)
  jac2 = compute_jacobian(x2,params,buffers,fnet)

  result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
  k = result.sum(0) 

  # reshape
  k=k.reshape(k.shape[0],k.shape[1])

  return k,jac2


def compute_kxkny(X_train,X_test,y_train,params,buffers,fnet):

  # compute kernels
  kn,jac_train=compute_kernel(X_train,X_train,params,buffers,fnet)
  kx,jac_test=compute_kernel(X_train,X_test,params,buffers,fnet)

  y_train=y_train.reshape(y_train.shape[0],1)


  # to avoid the singular matrix problem we add a gaussian noise when the det of the matrix is 0
  
  # if torch.linalg.det(kn)==0:
  #   kn=kn+torch.normal(0,0.01,kn.shape).to(device)

  kny= torch.stack([torch.einsum('ij,jk->ik', torch.linalg.inv(kn), y_train)])
  kny=kny.reshape(kny.shape[1],kny.shape[2])

  y_pred_tmp=torch.stack([torch.einsum('ij,jk->ik',kx.T, kny)])
  y_pred_tmp=y_pred_tmp.reshape(y_pred_tmp.shape[1],y_pred_tmp.shape[2])

  # compute the constant
  y_0=fnet(params, buffers, X_test)
  flatten_params=[p.flatten() for p in params]
  
  result = torch.stack([torch.einsum('Naf,f->Na', j1, j2) for j1, j2 in zip(jac_test, flatten_params)])
  k = result.sum(0)
  y_pred=y_pred_tmp+y_0-k

  return y_pred