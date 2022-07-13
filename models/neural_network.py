import torch
import torch.nn as nn

class NN(nn.Module):

    def __init__(self,input_size,output_size,width):

        super(NN, self).__init__()
        self.layernorm=nn.LayerNorm(input_size)
        self.linear1=nn.Linear(input_size, width)
        self.linear2=nn.Linear(width, width)

        # self.linear2=nn.Linear(width, width*2)
        # self.linear3=nn.Linear(width*2, width*3)
        self.linear4=nn.Linear(width, output_size)


    def forward(self, x):

        x = self.layernorm(x)

        x = self.linear1(x)
        x = nn.functional.relu(x)

        x = self.linear2(x)
        x = nn.functional.relu(x)

        x = self.linear4(x)

        return x



# class NN(nn.Module):

#     def __init__(self,input_size,output_size,width):

#         super(NN, self).__init__()
#         self.layernorm=nn.LayerNorm(input_size)
#         self.linear1=nn.Linear(input_size, width)
#         self.linear2=nn.Linear(width, 2*width)
#         self.linear3=nn.Linear(2*width, 4*width)
#         self.linear4=nn.Linear(4*width, 8*width)

#         # self.linear2=nn.Linear(width, width*2)
#         # self.linear3=nn.Linear(width*2, width*3)
#         self.linear5=nn.Linear(8*width, output_size)


#     def forward(self, x):

#         x = self.layernorm(x)

#         x = self.linear1(x)
#         x = nn.functional.relu(x)

#         x = self.linear2(x)
#         x = nn.functional.relu(x)

#         x = self.linear3(x)
#         x = nn.functional.relu(x)

#         x = self.linear4(x)
#         x = nn.functional.relu(x)

#         x = self.linear5(x)

#         return x