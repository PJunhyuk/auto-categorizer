import sys
from sklearn.externals import  joblib
from sklearn.grid_search import GridSearchCV
from sklearn.svm import  LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer,CountVectorizer
import os
import numpy as np
import string
from keras import backend
from keras.layers import Dense, Input, Lambda, LSTM, TimeDistributed
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model

### 파일에서 학습 데이터를 읽는다.

import json

x_text_list = []
y_text_list = []
enc = sys.getdefaultencoding()
with open("refined_category_dataset.dat",encoding=enc) as fin:
    for line in fin.readlines():
        # print (line)
        info = json.loads(line.strip())
        x_text_list.append((info['pid'],info['name']))
        y_text_list.append(info['cate'])

### text 형식으로 되어 있는 카테고리 명을 숫자 id 형태로 변환한다.

y_name_id_dict = joblib.load("y_name_id_dict.dat")

print(y_name_id_dict)

# y_name_set = set(y_text_list)
# y_name_id_dict = dict(zip(y_name_set, range(len(y_name_set))))
# print(y_name_id_dict.items())
# y_id_name_dict = dict(zip(range(len(y_name_set)),y_name_set))
y_id_list = [y_name_id_dict[x] for x in y_text_list]

### text 형태로 되어 있는 상품명을 각 단어 id 형태로 변환한다.

from sklearn.model_selection import train_test_split

vectorizer = CountVectorizer()
x_list = vectorizer.fit_transform(map(lambda i : i[1],x_text_list))
y_list = [y_name_id_dict[x] for x in y_text_list]

### train test 분리하는 방법

print(x_text_list[2])

X_train, X_test , y_train, y_test = train_test_split(x_text_list, y_list, test_size=0.2, random_state=42)

# print(vectorizer.transform(list(map(lambda i : i[1],X_train))))

### 몇개의 파라미터로 간단히 테스트 하는 방법

for c in [1,5,10]:
    clf = LinearSVC(C=c)
    X_train_text = map(lambda i : i[1],X_train)
    clf.fit(vectorizer.transform(X_train_text), y_train)
    print (c,clf.score(vectorizer.transform(map(lambda i : i[1],X_test)), y_test))

### 최적의 파라미터를 알아서 다 해보고, n-fold cross validation까지 해주는 방법 - GridSearchCV

svc_param = np.logspace(-1,1,4)

gsvc = GridSearchCV(LinearSVC(), param_grid= {'C': svc_param}, cv = 5, n_jobs = 4)

gsvc.fit(vectorizer.transform(map(lambda i : i[1],x_text_list)), y_list)

print(gsvc.best_score_, gsvc.best_params_)

#### 평가 데이터에 대해서 분류를 한 후에  평가 서버에 분류 결과 전송

eval_x_text_list = []
with open("soma8_test_data.dat",encoding=enc) as fin:
    for line in fin.readlines():
        info = json.loads(line.strip())
        eval_x_text_list.append((info['pid'],info['name']))

pred_list = clf.predict(vectorizer.transform(map(lambda i : i[1],eval_x_text_list)))

# print (pred_list.tolist())

import requests
name='박준혁'
nickname='jgravity_02'
mode='test'
param = {'pred_list':",".join(map(lambda i : str(int(i)),pred_list.tolist())),
         'name':name,'nickname':nickname,'mode':mode}
d = requests.post('http://eval.buzzni.net:20001/eval',data=param)

print (d.json())
