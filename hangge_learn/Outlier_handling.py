# coding:utf-8
# writer:zhouyuhang
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imchap3.imset import *
from pyod.models.iforest import IForest
from pyod.models.sod import SOD
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor

def by_3sigma(dataf,n_sigma=3):
    '''使用3sigma原则，找出异常值，并将其赋值为nan空值'''
    mean_data=dataf.mean()
    std_data=dataf.std()
    index=np.abs((dataf-mean_data)/std_data)>n_sigma*std_data
    try:
        dataf[index]=np.NaN
    except ValueError:
        print('by_3sigma没有发现异常值！')
    return dataf

def by_box(ser,n_per=1.5):
    '''使用箱线图进行异常值发现,其中ser为series类型'''
    q25=ser.quantile(q=0.25)
    q75=ser.quantile(q=0.75)
    iqr=q75-q25
    up=q75+n_per*iqr
    down=q25-n_per*iqr
    ser[ser>up]=np.NaN
    ser[ser<down]=np.NaN
    return ser

def by_DBSCAN_df(dataf,minsam=5,eps=10000):
    '''使用基于密度的DBSCAN算法发现dataf中的异常值,应当注意，这个函数是基于样本的异常值检测，
    即:异常样本检测。
    这里，将异常样本的所有特征都赋值为nan，然后通过其他方法填充之。然而总感觉过于暴力。
    其中eps是两个样本之间的最大距离'''
    outlier_dection=DBSCAN(min_samples=minsam,eps=eps)
    clusters=outlier_dection.fit_predict(dataf) # clusters为样本所属的类的标签，其中-1表示为异常值
    yichang_num=list(clusters).count(-1)
    index=np.array(clusters)==-1
    print('异常样本原索引: ',np.arange(len(dataf))[index])
    dataf.iloc[index,:]=np.NaN
    print('DESCAN异常值在min_samples为%s且eps为%s时，异常样本占比:'%(minsam,eps),yichang_num/len(dataf))
    print('共有%s个样本' % (len(dataf)))
    print('异常值个数: ',yichang_num)
    return dataf

def by_DESCAN_ser(series,min_sam=5,eps=100):
    '''这个函数与上述函数by_DESCAN_df不同，
    这个函数接受的是一个series类型，即一个字段，
    这样做的好处是，可以对单个特征进行异常值发现与处理，
    而不是像上述那样，粗暴的将整个样本删除'''
    series=by_DBSCAN_df(pd.DataFrame(series),minsam=min_sam,eps=eps)    # 直接将series转换成df，然后调用上述函数即可
    return series

def by_kMeans(dataf,n_clu=2,max_iter=1000):
    '''通过k均值聚类，如果某一个点不属于任何簇，则认为其是异常值。
    其中，n_clu是聚类簇数，max_iter是最大迭代次数,
    如果n_clusters为2，则一类为异常的，另一类为非异常的，
    同样地，这个函数是对整个样本而非某一字段作用的。'''
    data_scale=scale(dataf) # 标准化，但是需要注意返回的是ndarray类型
    model=KMeans(n_clusters=n_clu,max_iter=max_iter)
    model.fit(data_scale)
    data_scale=pd.DataFrame(data_scale)     # 转换为dataframe类型
    data_scale['labels']=model.labels_  # 添加标签列
    a,b=np.unique(model.labels_,return_counts=True)

    yichang_label=min(b)
    yichang_num=list(model.labels_).count(yichang_label)    # 异常值个数
    print('在kmeans的聚类簇数为%s且最大迭代次数为%s时，异常样本占比: '%(n_clu,max_iter),yichang_num/len(dataf))
    print('共有%s个样本'%len(dataf))
    print('异常值个数%s'%yichang_num)
    dataf[data_scale.labels==yichang_label]=np.NaN
    return dataf
    # 不好用！

def by_kMeans_ser(ser,n_clu=2,max_iter=1000):
    '''上述by_kMeans_ser的series形式'''
    ser_scale=(ser-ser.mean())/ser.std()
    model = KMeans(n_clusters=n_clu, max_iter=max_iter)
    model.fit(ser_scale.values.reshape(-1,1))
    ser=pd.DataFrame(ser)
    ser['labels']=model.labels_
    a,b=np.unique(model.labels_,return_counts=True)
    print(a)
    print(b)
    yichang_label=min(b)    # 异常值所在的标签
    yichang_num=list(model.labels_).count(yichang_label)
    print('在kmeans的聚类簇数为%s且最大迭代次数为%s时，异常样本占比: ' % (n_clu, max_iter), yichang_num / len(ser))
    print('共有%s个样本' % len(ser))
    print('异常值个数%s' % yichang_num)
    ser[ser['labels']==yichang_label]=np.NaN
    return ser.drop('labels',axis=1)
    # 效果仍然不佳

def by_iforest(dataf,n_esti=100,con=0.01,max_fea=10):
    '''使用孤立森林算法建立异常值检测模型，
    其中，n_esti是弱学习器的数量，
    con为指定的异常值所占比例，默认为1%
    max_fea为每个弱学习器最多可使用全部的特征数,
    缺点是：需要人为指定异常值所占比例；不能使用series类型。'''
    ifod=IForest(n_estimators=n_esti,
                 contamination=con,     # 指定异常值比例
                 max_features=max_fea,
                 random_state=123)
    ifod.fit(dataf)
    yichang_label=ifod.labels_  # 异常值标签,只含有0,1 其中标签为1代表异常值
    a,b=np.unique(yichang_label,return_counts=True)
    print('孤立森林在n_esti为%s且max_fea为%s时，选择比例为%s的结果:'%(n_esti,max_fea,con))
    print('共有%s个样本'%len(dataf))
    print('异常值的个数为：',b[1])
    dataf[yichang_label==1]=np.NaN
    return dataf

def by_sod(dataf,n_nei=20,ref_set=10,alpha=0.85,con=0.01):
    '''使用子空间异常值检测算法sod，
    其中，n_nei:k近邻查询异常值近邻数量，
    ref_set:创建参考集的共享最近邻数量。
    alpha:选择指定子空间的下限。
    con:异常值比例。'''
    sod=SOD(n_neighbors=n_nei,ref_set=ref_set,alpha=alpha,contamination=con)
    sod.fit(dataf)
    yichang_label=sod.labels_   # 标签仅有0,1 两种，1代表异常值标签
    print('sod算法在n_nei为%s时且ref_set为%s时且alpha为%s时，比例为%s选择异常值：'%(n_nei,ref_set,alpha,con))
    dataf[yichang_label==1]=np.NaN
    return dataf

def by_sod_ser(ser,n_nei=20,ref_set=10,alpha=0.85,con=0.01):
    sod = SOD(n_neighbors=n_nei, ref_set=ref_set, alpha=alpha, contamination=con)
    sod.fit(pd.DataFrame(ser))
    yichang_label = sod.labels_  # 标签仅有0,1 两种，1代表异常值标签
    print('sod算法在n_nei为%s时且ref_set为%s时且alpha为%s时，比例为%s选择异常值：' % (n_nei, ref_set, alpha, con))
    ser[yichang_label==1]=np.NaN
    return ser

# by_hnn的子函数
def get_index(arr,now_index,h=3):
    '''输入序列和近邻数和位置，返回其2h个近邻的索引'''
    index=np.array([1])
    if now_index<h:    # 如果索引过小
        index=np.arange(2*h)
    if now_index>=h and now_index<=len(arr)-1-h:    # 中间的位置
        index=np.append(np.arange(now_index-h,now_index),np.arange(now_index,now_index+h))
    if now_index>len(arr)-1-h:
        index=np.arange(len(arr)-2*h,len(arr))
    return index

def by_hnn(arr,n,h=4,remind=False):
    '''基于距离的异常值检测方法，
    其中，arr是输入ndarray数组，
    h为近邻个数（前后各有h个），
    n为控制阈值的量'''
    arr1=arr.copy()
    count=0
    for ii,num in enumerate(arr1):
        index=get_index(arr1,ii,h=h) # 获取邻居索引
        neighbors=arr1[index]    # 邻居
        nei_mean=neighbors.mean()   # 均值
        nei_std=neighbors.std() # 标准差

        if arr1[ii]<=nei_mean-n*nei_std or arr1[ii]>=nei_mean+n*nei_std:
            arr1[ii]=np.NaN
            count+=1
            if remind:
                print('发现异常值%s！已将其置为空值！'%num)
                print('共发现异常值个数: ',count)
            else:
                print('未发现异常值')
            
    return arr1

def by_k(arr,yu,n=2,h=4,kua=1,remind=False):
    '''通过斜率来去除特别尖的毛刺
    其中，arr为输入的ndarray
    yu为斜率的最小阈值
    n为n*sigma
    h为邻居个数
    kua为跨越的点的个数
    '''
    arr1=arr.copy()
    count=0
    for ii in np.arange(kua,len(arr)-kua):
        nei_index = get_index(arr,ii,h=h)
        nei=arr1[nei_index] # 邻居
        nei_mean=nei.mean()
        nei_std=nei.std()
        if ii==len(arr):
            pass
        conditions=[
            (arr1[ii+kua]-arr1[ii])*(arr1[ii]-arr1[ii-kua])<yu,
            arr1[ii]<nei_mean-n*nei_std or arr1[ii]>nei_mean+n*nei_std
        ]   # 条件列表
        if all(conditions):
            # 同时满足斜率条件和距离条件
            count+=1
            arr1[ii]=np.NaN
            if remind:
                print('发现异常值%s,已将其置为空值！'%arr1[ii])
                print('共发现异常值个数: ',count)
            else:
                print('未发现异常值')
    return arr1

def by_LOF(arr,n_nei=10,remind=False,plot=False):
    '''基于lof算法,arr为输入的ndarray类型'''
    arr1=arr.copy()
    lof=LocalOutlierFactor(n_neighbors=n_nei,metric='minkowski')
    # 作用于数据集
    out_index=lof.fit_predict(arr1)  # 每个值对应的标签，标签为-1为异常值，越接近-1位异常值的可能性越大
    arr1[out_index==-1]=np.NaN
    if remind:
        print('异常值的索引为: ',np.arange(len(arr))[out_index==-1])
        print('检测出异常值的数量为: ',np.sum(out_index==-1))
    if plot:
        out_factor=lof.negative_outlier_factor_
        # 将得分标准化
        radius=(out_factor.max()-out_factor)/(out_factor.max()-out_factor.min())
        plt.figure(figsize=(16,6))
        plt.plot(arr,'-',alpha=0.8)
        plt.scatter(np.arange(len(arr)),arr,s=800*radius,edgecolors='r',
                    facecolors='none',
                    label='LOF得分')
        plt.legend()
        plt.grid()
        plt.title('lof异常值得分可视化')
        plt.show()
    return arr1