

# s1061443_AIdea



## Rank

![image-20210529164635654](/Users/peter0512lee/Library/Application Support/typora-user-images/image-20210529164635654.png)



## Code Description



### Data preprocessing



把train中乳量是空的刪掉

```python
train = train.dropna(subset=['11'])
train.reset_index(drop=True, inplace=True)
```

將乳牛的空值填入平均體重

```python
avg_weight = birth['6'].mean()
birth['6'] = birth['6'].fillna(birth['6'].mean())
```

train 合併 spec, 當年當月有病1, 沒病0, 新增health欄位

```python
from datetime import datetime
train['health'] = 0
test['health'] = 0
for i in range(len(spec)):
    ym = datetime.strptime(spec['4'][i], "%Y/%m/%d %H:%M")
    if len(train.index[train["5"] == spec["1"][i]]) > 0:
        for j in train.index[train["5"] == spec["1"][i]]:
            if train['2'][j] == ym.year and train['3'][j] == ym.month and train['4'][j]==spec['7'][i]:
                train['health'][j] = 1
    if len(test.index[test["5"] == spec["1"][i]]) > 0:
        for j in test.index[test["5"] == spec["1"][i]]:
            if test['2'][j] == ym.year and test['3'][j] == ym.month and test['4'][j]==spec['7'][i]:
                test['health'][j] = 1
```

新增weight欄位

```python
train['weight'] = np.nan
test['weight'] = np.nan
for i in range(len(birth)):
    if len(train.index[train['5'] == birth['1'][i]])>0:
        for j in train.index[train['5'] == birth['1'][i]]:
                train['weight'][j] = birth['6'][i]
    if len(test.index[test['5'] == birth['1'][i]])>0:
        for j in test.index[test['5'] == birth['1'][i]]:
                test['weight'][j] = birth['6'][i]
train['weight'] = train['weight'].fillna(avg_weight)
test['weight'] = test['weight'].fillna(avg_weight)
```

新增season欄位

```python
train['season'] = ""
for index, row in train.iterrows():
    if int(train['3'][index]) >= 3 and int(train['3'][index]) <= 5:
        train['season'][index] = 'Spring'
    elif int(train['3'][index]) >= 6 and int(train['3'][index]) <= 8:
        train['season'][index] = 'Summer'
    elif int(train['3'][index]) >= 9 and int(train['3'][index]) <= 11:
        train['season'][index] = 'Autumn'
    else:
        train['season'][index] = 'Winter'

test['season'] = ""
for index, row in test.iterrows():
    if int(test['3'][index]) >= 3 and int(test['3'][index]) <= 5:
        test['season'][index] = 'Spring'
    elif int(test['3'][index]) >= 6 and int(test['3'][index]) <= 8:
        test['season'][index] = 'Summer'
    elif int(test['3'][index]) >= 9 and int(test['3'][index]) <= 11:
        test['season'][index] = 'Autumn'
    else:
        test['season'][index] = 'Winter'
```



### DNN Model



把要 one hot 的類別轉換成數字

```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
all_data=pd.concat([new_train,new_test])
all_data['4'] = labelencoder.fit_transform(all_data['4'])
all_data['5'] = labelencoder.fit_transform(all_data['5'])
all_data['season'] = labelencoder.fit_transform(all_data['season'])
all_data['health'] = labelencoder.fit_transform(all_data['health'])
new_train = all_data[0:len(new_train)]
new_test = all_data[len(new_train)::]
all_data=pd.concat([new_train,new_test])
```

把要的類別轉換成 one hot

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(all_data)
X = enc.transform(new_train).toarray()
X_test = enc.transform(new_test).toarray()
print(X.shape, X_test.shape)			
```

train, test 切開

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
```

Define RMSE Loss

```python
from keras import backend as K
def rmse(y_pred, y_true):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
```

Model design

```python
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.optimizers import Adam

model=Sequential()
model.add(Dense(256, input_dim=3098, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss=rmse, optimizer="adam", metrics=[rmse])
```

Model summary

```python
model.summary()
```

![image-20210529171247477](/Users/peter0512lee/Library/Application Support/typora-user-images/image-20210529171247477.png)



## Authors

- [@peter0512lee](https://www.github.com/peter0512lee)

