# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

#提取12.18号加入购物车的数据作为预测结果！

# pandas读取商品子集（P）
train_item=pd.read_csv('../fresh_comp_offline/tianchi_fresh_comp_train_item.csv')
print('读取商品子集完成！')

# pandas读取用户商品交互数据（D）
train_user=pd.read_csv('../fresh_comp_offline/tianchi_fresh_comp_train_user.csv')
print('读取用户商品交互数据！')

# 根据用户的心理行为，前一天的购物车商品很有可能第二天就被购买，
# 所以我们直接提交12月18号一天的购物车（跟商品子集交）

# 筛选出behavior_type==3，即加入购物车数据
train_user=train_user[train_user["behavior_type"]==3]

# 筛选出12月18号一天的数据
import re
regex=re.compile(r'^2014-12-18+ \d+$')
def date(column):
    if re.match(regex,column['time']):
        date,hour=column['time'].split(' ')
        return date
    else:
        return 'null'
train_user['time']=train_user.apply(date,axis=1)
print('日期和小时分离完成！')

train_user=train_user[(train_user['time'] =='2014-12-18')]
print('筛选2014-12-18号数据完成！')

# 删除掉多余项
train_user=train_user.drop(['user_geohash'],axis=1)
train_user=train_user.drop(['item_category'],axis=1)
train_user=train_user.drop(['behavior_type'],axis=1)
train_user=train_user.drop(['time'],axis=1)
print('已删掉多余的项！')

# 生成sample_submission.csv文件，保存
train_user.to_csv('../fresh_comp_offline/tianchi_mobile_recommendation_predict.csv',index=False)
print('文件保存已完成！')