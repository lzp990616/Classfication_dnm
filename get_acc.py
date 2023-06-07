import os
import re
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

def get_acc_list(data_name):
    # data_name = "ResNet18_lr-0.001.log"
    basePath =  os.getcwd()
    data_path = os.path.join(basePath, data_name)
    "Accuracy:"
    train_pattern = r"Train:\n.+?Accuracy: (\d+.\d{4})% \n"
    test_pattern = r"Test:\n.+?Accuracy: (\d+.\d{4})% \n"

    with open(data_path, 'r') as f:
        text = f.read()
        x = re.findall(train_pattern, text)
        #print(x)
        # import pdb
        # pdb.set_trace()
        train_result = []
        for match in x: 
            train_result.append(float(match))
        print(train_result)
        print(max(train_result))

        x = re.findall(test_pattern, text)
        #print(x)
        test_result = []
        for match in x: 
            test_result.append(float(match))
        print(test_result)
        print(max(test_result))
    return train_result, test_result

def get_loss_list(data_name):
    # data_name = "ResNet18_lr-0.001.log"
    basePath =  os.getcwd()
    data_path = os.path.join(basePath, data_name)
    "Accuracy:"
    train_pattern = r"Train:\n.+?Accuracy: \d+.\d% \n.+? Avg loss: (\d+\.\d+),"
    test_pattern = r"Test:\n.+?Accuracy: \d+.\d% \n.+? Avg loss: (\d+\.\d+),"

    with open(data_path, 'r') as f:
        text = f.read()
        x = re.findall(train_pattern, text)
        #print(x)
        train_result = []
        for match in x: 
            train_result.append(float(match))
        print(train_result)
        print(min(train_result))

        x = re.findall(test_pattern, text)
        #print(x)
        test_result = []
        for match in x: 
            test_result.append(float(match))
        # print(test_result)
        # print(min(test_result))
    return train_result, test_result




def txt2data(filename):
    '''将txt文件的数字存入到python列表里'''
    with open(filename, 'r') as fobj:
        data = fobj.read().splitlines()
        return [float(i) for i in data]


def draw(train_loss, val_loss):
    epoch = np.linspace(1, 200,200)  # 定义周期数
    print(len(epoch))
    fig = plt.figure(figsize=(15, 7.5)) # 声明图框对象，图框大小
    ax = plt.axes() # 声明坐标轴
    import pdb
    pdb.set_trace()
    ax.plot(epoch, train_loss, label='Train Loss', linewidth=2.0, color='red') # 画折线图
    ax.plot(epoch, val_loss, label='Validation Loss',linewidth=2.0, color='blue')

    ax.set_xlim(0, 210) # 设置x轴范围
    ax.set_ylim(0, 2.2)
    ax.set_xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 30}) # 设置x轴标签及属性
    ax.set_ylabel('VGG Loss', fontdict={'family': 'Times New Roman', 'size': 30})
    plt.xticks(fontname='Times New Roman', fontsize=16) # 设置x轴数字属性
    plt.yticks(fontname='Times New Roman', fontsize=16)

    ax.legend(fancybox=True, framealpha=0.3, shadow=False, prop={'family': 'Times New Roman', 'size': 25}) # 设置图例

    plt.show()

def draw2(x, y):
    epoch = np.linspace(1, 200,200)  # 定义周期数
    print(len(epoch))
    fig = plt.figure(figsize=(10, 7.5)) # 声明图框对象，图框大小
    ax = plt.axes() # 声明坐标轴

    model = make_interp_spline(x, y)
    xs = np.linspace(1,200,200)
    ys = model(xs)


    # ax.plot(epoch, xs, label='Train Loss', linewidth=2.0, color='red') # 画折线图
    # ax.plot(epoch, ys, label='Validation Loss',linewidth=2.0, color='blue')
    
    ax.set_xlim(0, 110) # 设置x轴范围
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 30}) # 设置x轴标签及属性
    ax.set_ylabel('Dice Loss', fontdict={'family': 'Times New Roman', 'size': 30})

    plt.xticks(fontname='Times New Roman', fontsize=16) # 设置x轴数字属性
    plt.yticks(fontname='Times New Roman', fontsize=16)

    # ax.legend(fancybox=True, framealpha=0.3, shadow=False, prop={'family': 'Times New Roman', 'size': 25}) # 设置图例

    plt.show()


if __name__ == "__main__":
    # train_result, test_result = get_acc_list("ResNetDNM_lr-0.001.log")
    train_result2, test_result2 = get_acc_list("ResNet18_lr-0.001.log")
    # import pdb
    # pdb.set_trace()
    # draw(train_result, test_result)
    # draw(train_result2, test_result2)
    # train_result, test_result = get_acc_list("ResNetDNM_lr-0.001.log")
    # train_result, test_result = get_loss_list("ResNetDNM_lr-0.001.log")
