#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import csv

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array
    
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784
    #return trainData,trainLabel
    
def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))  #  data 28000*784
    #return testData
    
#result是结果列表 
#csvName是存放结果的csv文件名
def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        myWriter.writerow(["ImageId","Label"])
        index=0;
        for i in result:
            tmp=[]
            index=index+1
            tmp.append(index)
            #tmp.append(i)
            tmp.append(int(i))
            myWriter.writerow(tmp)
            

from sklearn.ensemble import RandomForestClassifier
def RFClassify(trainData,trainLabel,testData):
    nbClf=RandomForestClassifier(n_estimators=100)
    nbClf.fit(trainData,ravel(trainLabel))
    testLabel=nbClf.predict(testData)
    saveResult(testLabel,'RF_Result.csv')
    return testLabel


def digitRecognition():
    trainData,trainLabel=loadTrainData()
    testData=loadTestData()
    result=RFClassify(trainData,trainLabel,testData)   
    print "finish!"

digitRecognition()
