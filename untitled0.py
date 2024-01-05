# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:24:45 2024

@author: M172504
"""
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder,StandardScaler
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = ConfigProto()
# config.allow_soft_placement = True
# config.gpu_options.per_process_gpu_memory_fraction = 1 # 分配显存
# config.gpu_options.allow_growth = True # 按需分配显存
# session = InteractiveSession(config=config)
# import tensorflow as tf
import warnings
from keras.utils import np_utils
from feature_engine.encoding import MeanEncoder
warnings.filterwarnings('ignore')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


def training_perpare(data,category,encoder=None):
    # feature engineering
    data[['policy_csl_ll', 'policy_csl_ul']] = data['policy_csl'].str.split('/',n=1, expand=True).astype(int)
    data['policy_bind_date']=pd.to_datetime(data['policy_bind_date'])
    data['policy_bind_date_year']=data['policy_bind_date'].dt.year.astype('int64')
    data['policy_bind_date_month']=data['policy_bind_date'].dt.month.astype('int64')
    data['policy_bind_date_day']=data['policy_bind_date'].dt.day.astype('int64')
    data['incident_date']=pd.to_datetime(data['incident_date'])
    data['incident_date_year']=data['incident_date'].dt.year.astype('int64')
    data['incident_date_month']=data['incident_date'].dt.month.astype('int64')
    data['incident_date_day']=data['incident_date'].dt.day.astype('int64')
    data['auto_year']=data['auto_year'].astype('int64')
    data['car_age'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d %H:%M:%S').max().year - data['auto_year']
    data['incident_days'] = pd.to_datetime(data['incident_date'], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(data['policy_bind_date'], format='%Y-%m-%d %H:%M:%S')
    data['incident_days'] = data['incident_days'].astype('timedelta64[D]')
    data['policy_bind_year'] = pd.to_datetime(data['policy_bind_date'], format='%Y-%m-%d %H:%M:%S').dt.year
    data = data.drop(['policy_bind_date','incident_date'], axis=1)
    data['second_hand'] = data['policy_bind_year'] - data['auto_year']
    data['second_hand'] = data['second_hand'].map(lambda x: 0 if x>0 else 1)
    # data['incident_hour_of_the_day'] = data['incident_hour_of_the_day'].map(lambda x: 'midnight' if x>0 and x<=6 else 'morning' if x>6 and x<=11 else 'afternoon' if x>11 and x<=6 else 'night')
     
    object_cols = [c for c in data.select_dtypes(include=['object'])]
    float_cols = [c for c in data.select_dtypes(include=['float64','int64'])]
    float_cols = list(filter(lambda x: x not in ['policy_id','fraud'], float_cols))
    
    if category == 'train':
        # meancoder_lst = ['auto_make', 'auto_model', 'incident_city','insured_hobbies']
        # encoder = MeanEncoder(
        #     variables=meancoder_lst,
        #     ignore_format=True)

        # encoder.fit(data[object_cols+float_cols], data['fraud'])
        # output = encoder.transform(data[object_cols+float_cols])
        Y = data['fraud']
    else:
        # meancoder_lst = ['auto_make', 'auto_model', 'incident_city','insured_hobbies']
        # output = encoder.transform(data[object_cols+float_cols])
        # encoder = []
        Y = []
    
    output = data
    lb = LabelEncoder()
    for i in object_cols:
        output[i] = lb.fit_transform(output[i])
    
    Standard_scaler = StandardScaler()
    train_flt_cols = Standard_scaler.fit_transform(output[float_cols].values)
    fea1 = pd.DataFrame(train_flt_cols,columns=float_cols)
    X = pd.concat([fea1,output[object_cols]], axis=1)

    return X,Y,encoder
    # return output,Y,encoder

X,Y,encoder = training_perpare(train_df,'train')
X_,_,_ = training_perpare(test_df,'test')


from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,random_state=420,test_size = 0.2)

train_X = np.array(X_train)
train_Y = np.array(Y_train)
X_validation = np.array(X_validation)


#test_X = np.array(test_feature)
#test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)

# model = tf.keras.Sequential([
#         tf.keras.layers.Conv1D(filters=32, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape = (train_X.shape[1],1)),
#         tf.keras.layers.Conv1D(filters=64, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
#         tf.keras.layers.Conv1D(filters=128, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
#         tf.keras.layers.MaxPool1D(pool_size=(5,), strides=2, padding='same'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
#         tf.keras.layers.Dense(units=1024, activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
#         tf.keras.layers.Dense(units=2, activation='softmax')
# ])

# # model = tf.keras.Sequential()
# # model.add(tf.keras.layers.Conv1D(64, 15, strides=1, padding='SAME',input_shape=(560, 1), use_bias=False))
# # model.add(tf.keras.layers.ReLU())
# # model.add(tf.keras.layers.Conv1D(64, 3, strides=2,padding='SAME'))
# # model.add(tf.keras.layers.BatchNormalization())
# # model.add(tf.keras.layers.Dropout(0.5))
# # model.add(tf.keras.layers.Conv1D(64, 3, strides=2,padding='SAME'))
# # model.add(tf.keras.layers.BatchNormalization())
# # model.add(tf.keras.layers.LSTM(64, dropout=0.5, return_sequences=True))
# # model.add(tf.keras.layers.LSTM(32))
# # model.add(tf.keras.layers.Dropout(0.5))
# # model.add(tf.keras.layers.Dense(2, activation="softmax"))

# model.summary()
# model.compile(loss='sparse_categorical_crossentropy', 
#               optimizer='adam', 
#               metrics=['categorical_accuracy'])

# num_epochs = 100
# model.fit(train_X, 
#           train_Y,
#           batch_size=64,
#           epochs=num_epochs,
#           verbose=2)
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

cat = CatBoostRegressor(depth=7, 
                        l2_leaf_reg=1, 
                        learning_rate=0.01, 
                        border_count = 128, 
                        bagging_temperature = 0.8, 
                        n_estimators=400,
                        early_stopping_rounds=300, 
                        subsample = 0.85,
                        random_seed=1,
                        verbose = 0)


lgbr = LGBMRegressor(num_leaves=10, 
                        reg_alpha=0.01, 
                        reg_lambda=0.01, 
                        max_depth=6, 
                        learning_rate=0.005, 
                        min_child_samples=3, 
                        random_state=2022,
                        n_estimators=2000, 
                        subsample=1, 
                        colsample_bytree=1)

from sklearn.ensemble import VotingRegressor

rg_model = VotingRegressor([('lgb', lgbr), ('catboost', cat)],weights=[1,1],n_jobs=12)

rg_model.fit(X_train,Y_train)
y_pred = rg_model.predict(X_validation)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y_validation.tolist(),np.array(y_pred))
print("AUC值为:", auc)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(Y_validation, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LabelEncoder')
plt.legend(loc="lower right")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score

RFC = RandomForestClassifier(class_weight={1: 1.5}, max_depth=3, max_features='log2', n_estimators=500)
MD = RFC.fit(X_train,Y_train)

Rfclf_fea = pd.DataFrame(MD.feature_importances_)
Rfclf_fea["Feature"] = list(X_train) 
Rfclf_fea.sort_values(by=0, ascending=False)



# test_X = np.array(X_)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)


pre = rg_model.predict(X_)
data_test_price = pd.DataFrame(pre,columns = ['fraud'])
results = pd.concat([test_df['policy_id'],data_test_price],axis = 1)
submit_file_z_score = r'result_21.csv'
results.to_csv(submit_file_z_score,encoding='utf8',index=0)