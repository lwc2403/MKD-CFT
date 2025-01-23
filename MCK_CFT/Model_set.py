import torch
import torch.nn as nn
from CrossFusionTransformer import CrossFusionTransformer
from resnet import ResNet,Res_net_2d,ResNet18
from transformer import Transformer_m

class CFT(nn.Module):
    def __init__(self, args):
        super(CFT, self).__init__()
        self.num_Genus = 1
        self.num_actions = 10
        self.model = CrossFusionTransformer(args)
        self.dense1 = torch.nn.Linear(384, self.num_Genus)      
        self.dense2 = torch.nn.Linear(384, self.num_actions)
        self.softmax = torch.nn.LogSoftmax(dim=1)   
        
    def forward(self, x):
        x = self.model(x)
        gen_dense = self.dense1(x)
        act_dense = self.dense2(x)
        output_gen = self.softmax(gen_dense)
        output_act = self.softmax(act_dense)
        return output_gen,output_act 

class ResNet_1d(nn.Module):
    def __init__(self, args):
        super(ResNet_1d, self).__init__()
        self.num_Genus = 1
        self.num_actions = 10
        self.model = ResNet(args)
        self.dense1 = torch.nn.Linear(3200, self.num_Genus)       
        self.dense2 = torch.nn.Linear(3200, self.num_actions)
        self.softmax = torch.nn.LogSoftmax(dim=1)      
    def forward(self, x):
        x = self.model(x)
        gen_dense = self.dense1(x)
        act_dense = self.dense2(x)
        output_gen = self.softmax(gen_dense)
        output_act = self.softmax(act_dense)
        return output_gen,output_act 

class ResNet_18_1d(nn.Module):
    def __init__(self, args):
        super(ResNet_18_1d, self).__init__()
        self.num_Genus = 1
        self.num_actions = 10
        self.model = ResNet18(args)
        self.dense1 = torch.nn.Linear(512, self.num_actions)
        self.dense2 = torch.nn.Linear(512, self.num_Genus)
        self.softmax = torch.nn.LogSoftmax(dim=1)      
    def forward(self, x):
        x = self.model(x)
        gen_dense = self.dense1(x)
        act_dense = self.dense2(x)
        output_gen = self.softmax(gen_dense)
        output_act = self.softmax(act_dense)
        return output_gen,output_act 
    
class ResNet_2d(nn.Module):
    def __init__(self, args):
        super(ResNet_2d, self).__init__()
        self.num_Genus = 1
        self.num_actions = 10
        self.model = Res_net_2d(args)
        self.dense1 = torch.nn.Linear(512, self.num_Genus)      
        self.dense2 = torch.nn.Linear(512, self.num_actions)
        self.softmax = torch.nn.LogSoftmax(dim=1)      
    def forward(self, x):
        x = self.model(x)
        gen_dense = self.dense1(x)
        act_dense = self.dense2(x)
        output_gen = self.softmax(gen_dense)
        output_act = self.softmax(act_dense)
        return output_gen,output_act 
        
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.num_Genus = 1
        self.num_actions = 10
        self.model = Transformer_m(1024, 2, 8)
        self.dense1 = torch.nn.Linear(1024, self.num_Genus)       
        self.dense2 = torch.nn.Linear(1024, self.num_actions)
        self.softmax = torch.nn.LogSoftmax(dim=1)      
    def forward(self, x):
        dx = x.size(1)
        x = self.model(x)
        x = torch.div(torch.sum(x, dim=1).squeeze(dim=1), dx)
        gen_dense = self.dense1(x)
        act_dense = self.dense2(x)
        output_gen = self.softmax(gen_dense)
        output_act = self.softmax(act_dense)
        return output_gen,output_act 
