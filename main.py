#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:13:10 2017

@author: renzhengzhong
@project: IRIE final project

"""

'''文字資料匯入'''
def Login_Doc(path):
    with open(path,'r', encoding = 'utf-8') as fin:
        tmp = fin.read()
    return tmp
#___________________________________________________

'''訓練與測試資料匯入'''
def Login_Data(path):
    with open(path,'r', encoding = 'utf-8') as fin:
        tmp = fin.readlines()
    return tmp
#___________________________________________________

'''對segment 過的data進行初步清理'''
def tidy_doc(doc):
    doc = doc.replace('，',' ')
    doc = doc.replace('。」','\n')
    doc = doc.replace('！」','\n')
    doc = doc.replace('？」','\n')
    doc = doc.replace('。','\n')
    doc = doc.replace('？','\n')
    doc = doc.replace('！','\n')
    return doc
#___________________________________________________

'''把字詞與POS進行分隔，並用list存起來'''
import re
def split_word_pos(doc):
    filter_pos = re.compile(r'[^a-zA-Z0-9 ]') 
    doc_list = doc.split(' ')
    wd_pos_list=[]
    for each in doc_list:               
        pos = filter_pos.sub('',each)
        wd = each.replace('_'+pos,'')
        wd_pos_list.append([wd,pos])
    return doc_list,wd_pos_list
#___________________________________________________

'''將doc list 進行分句,利用標點符號'''
def word_to_sentence(segment):
    sentence_list = []
    wd = []
    pos = []
    for each in segment:
        if each[0] != '\n':
            wd.append(each[0])
            pos.append(each[1])
        else:
            sentence_list.append([wd,pos])
            wd = []
            pos = []
    
    return sentence_list
#___________________________________________________

'''把訓練與測試資料做成DF'''
import pandas as pd
import copy # 會使用到copy.deepcopy()
def To_DF(data_set):
    data = copy.deepcopy(data_set) #避免動到餵入的原始資料
    doc_dict={}
    data.pop(0)
    for line in data:
        doc = line.split('\t')
        doc[3] = doc[3].strip()
        doc_dict[doc[0]] = doc
    DF = pd.DataFrame.from_dict(doc_dict,orient='index')
    DF.columns = ['ID','Entity1','Entity2','Relation']
    del DF['ID']
    return DF
#___________________________________________________

'''輸入entity pair, 並且手刻規則,去擷取出一段文字'''
def extract_content(sent_list,entity1,entity2):
    
    #找出各自第一次出現的句子
    def first_appear(entity):
        for i in range(len(sent_list)):
            if entity in sent_list[i][0]:
                index = i
                break
            elif entity[0] in sent_list[i][0] and entity[1:] in sent_list[i][0]:
                index = i
                entity = entity[1:]
                break
            elif entity[1:] in sent_list[i][0]:
                index = i
                entity = entity[1:]
                break
        return entity,index
    
    entity1, index1 = first_appear(entity1)
    entity2, index2 = first_appear(entity2)
    entity_pair = [entity1,entity2]
    startpoint = min(index1,index2)-10  
    index = []
        

    for i in range(startpoint,len(sent_list)):
        if entity_pair[0] in sent_list[i][0] and entity_pair[1] in sent_list[i][0] : #當句
            index = i
            break
        elif i >= 2 and i <= len(sent_list)-3 and entity_pair[0] in sent_list[i-2:i+3][0] and entity_pair[1] in sent_list[i-2:i+3][0] : #前後兩句
                index = [i-2,i+3]
                break
        else:
            index = [index1, index2] #如果都沒有一起出現, 就抓各自第一次出現時的句子
    return index

#___________________________________________________

''' 將DF中加入先前所取出的關鍵段落, 並透過POS將不相干的字性給濾除'''
#from pandas import Series
import numpy as np

def relation_list(sentence_list,DF):
    
    nDF = np.array(DF)
    pos_set = ('VHC','VC','VH','VCL','VA','VAC','VE','V2','VJ','Na','Nb','Nc','Neu')
    extract = []
    content = []
    
    for each in nDF:
        tmp = extract_content(sentence_list,each[0],each[1]) #抓出關鍵段落的位置
        if type(tmp) == int:
            for i in range(0,len(sentence_list[tmp][0])):
                if sentence_list[tmp][1][i] in pos_set: # filter useless wds by pos
                    extract.append(sentence_list[tmp][0][i])
                else:
                    pass
        elif type(tmp) == list and len(tmp) >0:
            tmp.sort()
            for index_ in range(tmp[0],tmp[1]):
                for i in range(0,len(sentence_list[index_][0])):
                    if sentence_list[index_][1][i] in pos_set: # filter useless wds by pos
                        extract.append(sentence_list[index_][0][i])
                    else:
                        pass                         
        content.append(extract)
        extract = []
    content = np.array([DF['Relation'],content]).transpose()
    DF = pd.DataFrame(content,columns = ['Relation','Extraction'])
    #DF_ = pd.concat([content, DF],axis = 1)
    
    return DF  #這邊處理得不好 一定要再處理好一點
#___________________________________________________
#透過訓練資料建立namelist 清單
def nameList(DF):
    name_list = []
    DF = np.array(DF)
    for each in DF:
        name_list.append(each[0])
        name_list.append(each[1])
        if len(each[0]) > 2 :
            name_list.append(each[0][0])
            name_list.append(each[0][1:])
        if len(each[1]) > 2 :
            name_list.append(each[1][0])
            name_list.append(each[1][1:])
    name_list.extend(['王','薛','史','賈','林'])    
    name_list = list(set(name_list))
    return name_list
#___________________________________________________
'''''''''利用count來找feature'''''''''
def get_count_features(group_relation,name_list):
    group_relation = np.array(group_relation)
    relation_dict = {}

    for each in group_relation: #類別
        if each[0] not in relation_dict.keys():
            relation_dict[each[0]] = each[1]
        else:
            relation_dict[each[0]].append(each[1])
    features = {}   
    
    count = {}
    stop_list = ['道','說','一','有','人','來','去','家','聽','笑','好','看',\
                 '忙','心','到','出','兩','話','事','二','吃','問','大','鳳姐']
    stop_list.extend(name_list)
    for relation,content in relation_dict.items():
        for sent in content:
            for wd in sent:
                if wd not in count.keys() and wd not in stop_list:
                    count[wd] = 1
                elif wd in count.keys() and wd not in stop_list:
                    count[wd] = count[wd]+1
                else:
                    pass
        features[relation] = count
        count = {}  
    #將 features 依照個數做排序 並且只取前___個 當成該該relation的features
    Features = {}
    for ID, group in features.items():
        feature_sort = sorted(features[ID].items(), key=lambda x: x[1], reverse = True)
        Features[ID] = feature_sort
        feature_sort = []
        
    for ID, group in Features.items():
        try:
            Features[ID] = Features[ID][1:30]
        except:
            Features[ID] = Features[ID][1:]
        
    return Features

#___________________________________________________    
def get_feature_set(Features):
    feature_set = []
    for relation, feature in Features.items():
        for each_tuple in feature:
            feature_set.append(each_tuple[0])
    
    feature_set = list(set(feature_set))
    #feature_set = np.array(feature_set)
    return feature_set
#___________________________________________________    
'''透過餵入: bow 與 特徵集 --> 做出特徵矩陣'''
def feature_matrix(bow, Feature_set):
    #Bow = bow # DF_train
    bow = np.array(bow)
    DF = pd.DataFrame(columns = Feature_set)
    bool_matrix = []
    ID_count = 0
    for element in bow:
        for each_feature in Feature_set:        
            if each_feature in element[1] :
                bool_matrix.append(1)
            else:
                bool_matrix.append(0)
        DF.loc[ID_count ] =  bool_matrix
        bool_matrix = []
        ID_count = ID_count +1
    relation = [each[0] for each in bow]
    DF.insert(len(Feature_set),'Relation',relation)
    return DF

#___________________________________________________    
#import jieba.analyse
'''利用結巴套件來計算每種關係中，每個詞彙的tf-idf值,並取出前10大詞彙當成其特徵值'''
'''
def features(bow):
    Feature = {}

    for ID, test in bow.items():
        feature = jieba.analyse.extract_tags(bow[ID],10,withWeight = False,allowPOS = False)
        Feature[ID] = feature

    return Feature
#___________________________________________________    
'''

'''implement'''
#資料匯入
raw_data = Login_Doc("/Users/renzhengzhong/Desktop/IRIE/Project2/IRIE_Project_2_data/Dream_of_the_Red_Chamber.txt")
seg_data = Login_Doc("/Users/renzhengzhong/Desktop/IRIE/Project2/IRIE_Project_2_data/Dream_of_the_Red_Chamber_seg.txt")
train_data = Login_Data("/Users/renzhengzhong/Desktop/IRIE/Project2/IRIE_Project_2_data/train.txt")
test_data = Login_Data("/Users/renzhengzhong/Desktop/IRIE/Project2/IRIE_Project_2_data/test.txt")

#初步過濾(把句號、驚嘆號用'\n' 取代)
tidy_doc = tidy_doc(seg_data)

#將文字與標籤分開標示
doc_list,seg = split_word_pos(tidy_doc) #doc_list 目前沒作用

#把訓練與測試資料做成DataFrame的形式
DF_train = To_DF(train_data)
DF_test = To_DF(test_data)

#把單一wd串成一個較長的句子(list)
sent_list = word_to_sentence(seg)

train_bow = relation_list(sent_list,DF_train)
test_bow = relation_list(sent_list,DF_test)

word_counts = get_count_features(train_bow, nameList(DF_train))

features = get_feature_set(word_counts)

# 以訓練集所萃出出的特徵矩陣為最後的output變數
Feature_matrix = feature_matrix(train_bow, features)
# 計算測試集的特徵矩陣為何(將用於丟進機器學習)
test_matrix = feature_matrix(test_bow, features)

#___________________________________________________

'''訓練與測試資料的準備'''
from sklearn import  metrics   
# 建立訓練與測試資料
train_X = Feature_matrix.iloc[:,0:len(Feature_matrix.columns)-1]
train_y = Feature_matrix.iloc[:,90] #先寫死 待會改
test_X = test_matrix.iloc[:,0:len(test_matrix.columns)-1]
test_y = test_matrix.iloc[:,90]
#, test_X, train_y, test_y


''' SVM 分類器 '''
from sklearn import svm
# 建立向量支持器 分類器
# SVC參數kernel:它指定要在算法中使用的內核類型,
# 有:'linear','poly','rbf'(default),'sigmoid','precomputed'
svc = svm.SVC(kernel='rbf')
svc_fit = svc.fit(train_X, train_y)

# 預測
svc_test_y_predicted = svc.predict(test_X)
# 績效
svc_accuracy = metrics.accuracy_score(test_y, svc_test_y_predicted)
print('Accuracy for SVM = ',svc_accuracy) 

cm = metrics.confusion_matrix(test_y, svc_test_y_predicted)

#使用kernel='linear', 再加入辭典(.strip())
print('TRAIN SCORE: ',svc.score(train_X, train_y),' TEST SCORE: ', svc.score(test_X, test_y))
#__________________________________________________________________
from sklearn.ensemble import RandomForestClassifier
# 建立 random forest 分類器
forest = RandomForestClassifier(n_estimators = 10) #n_jobs=-1,max_features='auto', \
                                #n_estimators = 3,random_state = 0)

forest_fit = forest.fit(train_X, train_y)

# 預測
RanFor_test_y_predicted = forest.predict(test_X)
# 績效
RF_self = forest.predict(train_X)
cm_self = metrics.confusion_matrix(train_y, RF_self)
RanFor_accuracy = metrics.accuracy_score(test_y, RanFor_test_y_predicted)
cm = metrics.confusion_matrix(test_y, RanFor_test_y_predicted)
print('Accuracy for Random Forests  = ',RanFor_accuracy)  
print('TRAIN SCORE: ',forest.score(train_X, train_y),' TEST SCORE: ', forest.score(test_X, test_y))
