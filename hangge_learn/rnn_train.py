# 专门训练循环神经网络的函数
import torch
from torch import nn
import pandas as pd


def rnn_trainer(model, train_loader, test_loader, optimizer,
                epoch_num: int, loss_func, print_gap: int, device='gpu'):
    """
    对循环神经网络的训练函数。

    Parameters:

    model:待训练的循环神经网络
    train_loader:train_loader
    test_loader: test_loader
    optimizer:优化器,如: optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num: 迭代次数
    loss_func: 损失函数,如: nn.MSELoss()
    print_gap: 如果不为0,则每print_gap次迭代输出一次在训练集和验证集上的误差
    device: 使用gpu还是cpu
    return: lossList (在训练集上的损失变化列表),lossListTest (在测试集上的损失变化列表)
    """
    if device == 'gpu':
        device = try_gpu()
        model = model.to(device)

    lossList = []  # 记录训练loss
    lossListTest = []  # 记录测试loss

    for epoch in range(epoch_num):
        loss_nowEpoch = []
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)  # 模型输入
            # loss计算，将batch_y从(64,7,4)变形为(64,28)- 这个是out的形状
            Loss = loss_func(out, batch_y)
            optimizer.zero_grad()  # 当前batch的梯度不会再用到，所以清除梯度
            Loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            loss_nowEpoch.append(Loss.item())
            break
        lossList.append(sum(loss_nowEpoch)/len(loss_nowEpoch))

        loss_nowEpochTest = []
        model.eval()
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            Loss = loss_func(out, batch_y)  # 将batch_y从(64,7,4)变形为(64,28)
            loss_nowEpochTest.append(Loss.item())
            break
        lossListTest.append(sum(loss_nowEpochTest)/len(loss_nowEpochTest))

        if print_gap:
            if epoch % print_gap == 0:
                print(">>> EPOCH{} averTrainLoss:{:.3f} averTestLoss:{:.3f}".format(
                    epoch+1, lossList[-1], lossListTest[-1]))

    return lossList, lossListTest


def transformer_trainer(model, train_loader, test_loader, optimizer,
                epoch_num: int, loss_func, print_gap: int, device='gpu'):
    """
    对循环神经网络的训练函数。

    Parameters:

    model:待训练的循环神经网络
    train_loader:train_loader
    test_loader: test_loader
    optimizer:优化器,如: optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_num: 迭代次数
    loss_func: 损失函数,如: nn.MSELoss()
    print_gap: 如果不为0,则每print_gap次迭代输出一次在训练集和验证集上的误差
    device: 使用gpu还是cpu
    return: lossList (在训练集上的损失变化列表),lossListTest (在测试集上的损失变化列表)
    """
    if device == 'gpu':
        device = try_gpu()
        model = model.to(device)

    lossList = []  # 记录训练loss
    lossListTest = []  # 记录测试loss

    for epoch in range(epoch_num):
        loss_nowEpoch = []
        model.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x,batch_y)  # 模型输入
            # loss计算，将batch_y从(64,7,4)变形为(64,28)- 这个是out的形状
            Loss = loss_func(out, batch_y)
            optimizer.zero_grad()  # 当前batch的梯度不会再用到，所以清除梯度
            Loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            loss_nowEpoch.append(Loss.item())
            break
        lossList.append(sum(loss_nowEpoch)/len(loss_nowEpoch))

        loss_nowEpochTest = []
        model.eval()
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x,batch_y)
            Loss = loss_func(out, batch_y)  # 将batch_y从(64,7,4)变形为(64,28)
            loss_nowEpochTest.append(Loss.item())
            break
        lossListTest.append(sum(loss_nowEpochTest)/len(loss_nowEpochTest))

        if print_gap:
            if epoch % print_gap == 0:
                print(">>> EPOCH{} averTrainLoss:{:.3f} averTestLoss:{:.3f}".format(
                    epoch+1, lossList[-1], lossListTest[-1]))

    return lossList, lossListTest


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def create_lstm(history_feas: int, pre_days: int, pre_feas: int,
                hidden_size: int, numlayers: int = 2, is2gpu: bool = False):
    """
    返回一个创建好的LSTM对象。
    Parameters:

    history_feas:使用历史上几个特征? 也就是自然语言处理中的vocab_size.
    pre_days:要预测未来几天的数据?
    pre_feas:要预测未来数据的几个特征?
    hidden_size:LSTM隐藏神经元的个数,同时也是全连接层的第一层神经元的个数
    num_layers:LSTM的层数
    is2gpu:是否返回在gpu上的模型?默认为False。
    """
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.lstm = nn.LSTM(input_size=history_feas, hidden_size=hidden_size,
                                num_layers=numlayers, batch_first=True)
            # output_size=2，输出的特征的维度数。
            self.linear = nn.Linear(hidden_size, pre_feas)

        def forward(self, x):
            out, h = self.lstm(x, None)
            out = self.linear(out[:, -pre_days:, :])  # 这里是 -2: ，则会输出两个时间维度
            # out.shape:(batch_size,time_step,output_size)
            return out

    model = MyModel()
    if is2gpu:
        return model.to(try_gpu())
    return model


def get_pre_true_df(net, data_set):
    """ 
    使用net和train_data_set(test_data_set),返回两个DataFrame.
    其中第一个DataFrame的每个列是各个时刻的predict值。
    第二个DataFrame的每个列是各个时刻的真实值

    需确保pandas已导入。

    Parameters:

    net:训练好的循环神经网络
    data_set:torch的标准输入,如train_data_set,test_data_set均可
    """
    true_df = pd.DataFrame()
    pre_df = pd.DataFrame()

    # 获取时刻的个数(预测)
    for idx, (x, y) in enumerate(data_set):
        num = len(y.flatten())
        if idx == 0:
            break

    # 数据写入DataFrame
    for i in range(num):  # 构造DataFrame的每一个列
        true_list = []
        pre_list = []
        for x, y in data_set:
            pre_list.append(
                net(x.reshape(1, x.shape[0], x.shape[1])).flatten()[i].item())
            true_list.append(y.flatten()[i].item())
        true_df['true'+str(i)] = true_list
        pre_df['pre'+str(i)] = pre_list

    return pre_df, true_df


# 似乎上面这种方法太慢了因为有两个for循环,下面是一个更快的方法
def get_pre_true_df_fast(net, data_set):
    """ 
    使用net和train_data_set(test_data_set),返回两个DataFrame.
    其中第一个DataFrame的每个列是各个时刻的predict值。
    第二个DataFrame的每个列是各个时刻的真实值

    需确保pandas已导入。

    Parameters:

    net:训练好的循环神经网络
    data_set:torch的标准输入,如train_data_set,test_data_set均可
    """

    # 获取时刻的个数(预测)
    for idx, (x, y) in enumerate(data_set):
        num = len(y.flatten())
        if idx == 0:
            break
        
    pre_columns=["pre{}".format(i) for i in range(num)]
    true_columns=["true{}".format(i) for i in range(num)]
    
    pre_arr=[]  # 每一行是一个y数组,每一列对应一个时刻
    true_arr=[]  # 每一行是一个y数组,每一列对应一个时刻
    
    for x,y in data_set:
        pre=net(x.reshape(1,x.shape[0],x.shape[1])).flatten().detach().numpy()
        pre_arr.append(pre)
        true=y.flatten().detach().numpy()
        true_arr.append(true)
        
    pre_df=pd.DataFrame(pre_arr,columns=pre_columns)
    true_df=pd.DataFrame(true_arr,columns=true_columns)
    
    return pre_df,true_df


def get_overlap_pre_true_df(net,dataset,pre_days:int):
    """ 
    获取重叠的预测结果的true_df和pre_df
    其中,我们对history_data和pre_data的重叠部分不感兴趣,
    只对每一个X,y对中y超出X的部分感兴趣(这相对于X是未来的数据).
    
    Parameters:
    
    - net:训练好的网络
    - dataset:train_dataset or test_dataset
    - pre_days:每一个X,y对中y超出X的天数,也就是未来的时刻个数
    
    return:
    重叠的预测结果的true_df和pre_df
    """
    pre_columns=["pre{}".format(i) for i in range(pre_days)]
    true_colummns=["true{}".format(i) for i in range(pre_days)]
    # pre_df=pd.DataFrame(columns=pre_columns)
    # true_df=pd.DataFrame(columns=true_colummns)
    pre_arr=[]  # 每一行是一个y数组,每一列对应一个时刻
    true_arr = []  # 每一行是一个y数组,每一列对应一个时刻
    
    for x,y in dataset:
        pre=net(x.reshape(1,x.shape[0],x.shape[1])).flatten().detach().numpy()[-pre_days:]
        pre_arr.append(pre)
        true=y.flatten().numpy()[-pre_days:]  # 别忘了加上负号,表示从第倒数pre_days个数开始,一直到数组的最后
        true_arr.append(true)
        
    pre_df=pd.DataFrame(pre_arr,columns=pre_columns)
    true_df=pd.DataFrame(true_arr,columns=true_colummns)
    return pre_df,true_df