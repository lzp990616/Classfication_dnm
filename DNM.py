import pdb
import re
import torch
from torch import nn
import torch.nn.functional as F

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
        k = torch.tensor(k).cuda()
        qs = torch.tensor(qs).cuda()

        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.k = k
        self.qs = qs
        self.activation = activation

    def forward(self, x):
        # Synapse
        # pdb.set_trace()
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

class DNM_multiple(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.5, qs=0.1):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.DNM_Linear1 = DNM_Linear(input_size, 320, M, k, qs)
        self.DNM_Linear2 = DNM_Linear(320, 50, M, k, qs)
        self.DNM_Linear3 = DNM_Linear(50, out_size, M, k, qs)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        x = self.DNM_Linear2(x)
        x = self.DNM_Linear3(x)
        out = F.softmax(x, dim=1)

        return out

class DNM_CNN(nn.Module):
    def __init__(self, num_classes, M, k=0.5, qs=0.1):
        super(DNM_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3),
                                         torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3))
        
        self.DNM_Linear = DNM_Linear(7*7*20, num_classes, M, k, qs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*20)
        x = self.DNM_Linear(x)
        out = F.softmax(x, dim=1)

        return out

class DNM_CNN(nn.Module):
    def __init__(self, num_classes, M, k=0.5, qs=0.1):
        super(DNM_CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3),
                                         torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3))
        
        self.DNM_Linear = DNM_Linear(7*7*20, num_classes, M, k, qs)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*20)
        x = self.DNM_Linear(x)
        out = F.softmax(x, dim=1)

        return out

class CNN(nn.Module):
    def __init__(self, out_size):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,10,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3),
                                         torch.nn.Conv2d(10,20,kernel_size=3,stride=1,padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=3))
        
        self.linear = nn.Linear(7*7*20, out_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*20)
        x = self.linear(x)
        out = F.softmax(x, dim=1)

        return out


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 32)
        self.l2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = self.l2(x)
        return x


class Vgg16_net(nn.Module):
    def __init__(self, num_classes):
        super(Vgg16_net, self).__init__()


        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            #inplace-选择是否进行覆盖运算
            #意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            #意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            #这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )


        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )


        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc=nn.Sequential(
            #y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            #nn.Liner(in_features,out_features,bias)
            #in_features:输入x的列数  输入数据:[batchsize,in_features]
            #out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            #bias: bool  默认为True
            #线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024,num_classes)
        )


    def forward(self,x):
        x=self.conv(x)
        #这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        #如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 512*7*7)
        x=self.fc(x)
        return x


class Vgg16_DNM(nn.Module):
    def __init__(self, num_classes, M, k=0.5, qs=0.1):
        super(Vgg16_DNM, self).__init__()


        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            #inplace-选择是否进行覆盖运算
            #意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            #意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            #这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )


        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )


        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.DNM_Linear1 = DNM_Linear(1024,num_classes, M, k, qs)

        self.fc=nn.Sequential(
            #y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            #nn.Liner(in_features,out_features,bias)
            #in_features:输入x的列数  输入数据:[batchsize,in_features]
            #out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            #bias: bool  默认为True
            #线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            DNM_Linear(1024,num_classes, M, k, qs, activation=None)
        )


    def forward(self,x):
        x=self.conv(x)
        x = x.view(-1, 512*7*7)
        x=self.fc(x)
        
        return x