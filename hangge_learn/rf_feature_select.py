# coding:utf-8
# writer:zhouyuhang

def rfr_select(xx,yy,n=10,printdf=False,df_tidy=False):
    '''随机森林选择回归模型的特征。返回表头的ndarray
    其中：xx为输入dataframe,
    yy为输出dataframe,
    n为返回的输入特征的重要性排序前n个'''
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    rfr=RandomForestRegressor(random_state=123)
    rfr.fit(xx,yy)
    fea=rfr.feature_importances_    # 各个特征的重要性
    index=fea.argsort()[::-1]   # 从大到小排序的索引
    return_fea=xx.columns.values[index]     # 排序好的特征名称
    aa=fea[index]   # 排序好的特征重要性得分。
    df=pd.DataFrame({'feature':return_fea[:n],'importances':aa[:n]})
    if printdf==True:
        print(df)
    if df_tidy==True:
        df
    return return_fea[:n]

def rfc_select(xx,yy,n=10,printdf=False,df_tidy=False):
    '''随机森林选择分类模型的特征。参数同rfr_select,返回表头的ndarray'''
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    rfc=RandomForestClassifier(random_state=123)
    rfc.fit(xx,yy)
    fea = rfc.feature_importances_  # 各个特征的重要性
    index = fea.argsort()[::-1]  # 从大到小排序的索引
    return_fea = xx.columns.values[index]  # 排序好的特征名称
    aa = fea[index]  # 排序好的特征重要性得分。
    df = pd.DataFrame({'feature': return_fea[:n], 'importances': aa[:n]})
    if printdf == True:
        print(df)
    if df_tidy == True:
        df
    return return_fea[:n]