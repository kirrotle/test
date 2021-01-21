import os
import random
import csv
import math
import numpy
from keras.models import Sequential
from keras.layers import Dense,Dropout
from data import data
from data_item import data_item



class model:
    def __init__(self):
        self.item=data().data 
        train_input,train_output,test_input,test_output=self.data_process(self.item)
        predict=self.model(train_input,train_output,test_input)
        result,difficult=self.reduction(predict,test_output)
        self.confuse_matrix(result,difficult)

    def data_process(self,item):
        train_input=[]
        train_output=[]
        test_input=[]
        test_output=[]

        for key in  item.keys():
            train_input.append(item[key].bert_tranceform)
            train_output.append(item[key].difficult/6)
            test_input.append(item[key].bert_tranceform)
            test_output.append(item[key].difficult/6)
        return train_input,train_output,test_input,test_output

    def model(self,train_input,train_output,test_input):

        model=Sequential()

        model.add(Dense(use_bias=1,input_dim=len(train_input[0]),
                        units=30,
                        activation='relu'))

        model.add(Dropout(0.25))
        model.add(Dense(use_bias=1,units=20,activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(use_bias=1,units=10,activation='relu'))       
        model.add(Dropout(0.25))
        model.add(Dense(use_bias=1,units=1,activation='relu'))      
        model.compile(loss='mse', 
              optimizer='adam',
              metrics=['mse'])

        model.fit(train_input,train_output,batch_size=16,epochs=100,verbose=1)

        predict=model.predict(test_input)

        return predict

    def reduction(self,predict,difficult):
        for count in range(len(difficult)):
            difficult[count]=int(round((difficult[count])*6,1))
        
        result=[]
        for count in range(len(predict)):
            if predict[count][0]<=(1/6):
                result.append(1)
            elif predict[count][0]<=(2/6):
                result.append(2)
            elif predict[count][0]<=(3/6):
                result.append(3)
            elif predict[count][0]<=(4/6):
                result.append(4)
            elif predict[count][0]<=(5/6):
                result.append(5)
            else:
                result.append(6)
        
        assert len(result)==len(difficult),"還原難度錯誤"
        
        return result,difficult

    def confuse_matrix(self,result,test_output):
        matrix=[[0 for i in range(6)] for i in range(6)]
        for count in range(120):
            matrix[test_output[count]-1][result[count]-1]+=1

        print(matrix)
    
model()
