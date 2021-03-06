# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

#提取12.10-18日的加入购买车数据作为预测结果！

# pandas读取商品子集（P）
train_item=pd.read_csv('../fresh_comp_offline/tianchi_fresh_comp_train_item.csv')
print('读取商品子集完成！')

# pandas读取用户商品交互数据（D）
train_user=pd.read_csv('../fresh_comp_offline/tianchi_fresh_comp_train_user.csv')
print('读取用户商品交互数据！')


# 筛选出behavior_type==3，即加入购物车数据
train_user=train_user[train_user["behavior_type"]==3]

# 筛选出12月10-18号的数据
import re
regex=re.compile(r'^2014-12-1+ \d+$')
def date(column):
    if re.match(regex,column['time']):
        date,hour=column['time'].split(' ')
        return date[:-1]+'0'
    else:
        return 'null'
train_user['time']=train_user.apply(date,axis=1)
print('日期和小时分离完成！')

train_user=train_user[(train_user['time'] =='2014-12-10')]
print('筛选2014-12-10至18号数据完成！')

# 删除掉多余项
train_user=train_user.drop(['user_geohash'],axis=1)
train_user=train_user.drop(['item_category'],axis=1)
train_user=train_user.drop(['behavior_type'],axis=1)
train_user=train_user.drop(['time'],axis=1)
print('已删掉多余的项！')

# 生成sample_submission.csv文件，保存
train_user.to_csv('../fresh_comp_offline/tianchi_mobile_recommendation_predict.csv',index=False)
print('文件保存已完成！')