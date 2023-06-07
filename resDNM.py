import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Soma(nn.Module):
    def __init__(self, k, qs):
        super(Soma, self).__init__()
        self.params = nn.ParameterDict({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.params['k'] * (x - self.params['qs'])))
        return y

class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 1)
        return x
        
class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.prod(x, 2)
        return x

class Synapse(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'k': nn.Parameter(k)})

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))
        y = 1 / (1 + torch.exp(
            torch.mul(-self.params['k'], (torch.mul(x, self.params['w']) - self.params['q']))))

        return y



class BASE_DNM(nn.Module):
    def __init__(self, dim, M, kv=5, qv=0.3):#, device=torch.device('cuda:0')):
        
        w = torch.rand([M, dim])#.to(device)
        q = torch.rand([M, dim])#.to(device)
        #k = torch.tensor(kv)
        #qs = torch.tensor(qv)
        k = torch.rand(1)
        qs = torch.rand(1)

        super(BASE_DNM, self).__init__()
        self.model = nn.Sequential(
            Synapse(w, q, k),
            Dendritic(),
            Membrane(),
            Soma(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        return x



class DNM_Linear(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1, activation=F.relu):
        super(DNM_Linear, self).__init__()

        DNM_W = torch.rand([out_size, M, input_size])#.cuda() # [size_out, M, size_in]
        dendritic_W = torch.rand([input_size])#.cuda() # size_out, M, size_in]
        membrane_W = torch.rand([M])#.cuda() # size_out, M, size_in]
        q = torch.rand([out_size, M, input_size])#.cuda()
        torch.nn.init.constant_(q, qs) # 设置q的初始值
        k = torch.tensor(k).to(device)
        qs = torch.tensor(qs).to(device)

        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.activation = activation

    def forward(self, x):
        # Synapse
        pdb.set_trace()
        out_size, M, _ = self.params['DNM_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.relu(torch.mul(self.k, (torch.mul(x, self.params['DNM_W']) - self.params['q'])))

        # x = torch.mul(x, self.params['DNM_W'])
        # x = F.relu(self.k * (x - self.params['q']))

        # Dendritic
        #x = torch.mul(x, self.params['dendritic_W'])
        #x = x * self.params['dendritic_W']
        x = torch.sum(x, 3)
        #x = torch.sigmoid(x)
        x = F.relu(x)

        # Membrane
        #x = torch.mul(x, self.params['membrane_W'])
        #x = x * self.params['membrane_W']
        x = torch.sum(x, 2)

        # Soma
        if self.activation != None:
            x = self.activation(self.k * (x - self.qs))

        return x

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512*3*3, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNetDNMBase(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNetDNMBase, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc1 = nn.Linear(512*3*3, 1024)
        self.DNM_Linear1 = DNM_Linear(512*3*3, num_classes,10,activation=None)
        self.DNM_Linear2 = DNM_Linear(1024, num_classes,10,activation=None)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #
        out = F.avg_pool2d(out, 4)
        pdb.set_trace()
        out = out.view(out.size(0), -1)
        #out = self.fc(out)
        # out = F.relu(self.fc1(out))
        # out = self.DNM_Linear2(out)
        out = self.DNM_Linear1(out)
        return out

def ResNetDNM(num_classes=2):
    return ResNetDNMBase(ResidualBlock, num_classes=num_classes)

def ResNet18(num_classes=2):
    return ResNetDNMBase(ResidualBlock, num_classes=num_classes)