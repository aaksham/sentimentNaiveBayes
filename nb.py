#importing libraries
import os
import re
import nltk
import time
import numpy
import math
from collections import defaultdict

###################################################
#               Global Variables
###################################################
dictionary=defaultdict(list)
pattern="\w+(?:[-']\w+)*|'|[-.(]+|\S\w*"
matrix=[]
labels=[]
r=0
###################################################
#           Helper Functions
###################################################
def populate_list(which):
    f=open(os.getcwd()+r'\vocab\\'+which+'-words.txt','r')
    word_list=f.read().strip().split('\n')
    for line in word_list:
        if len(line)==0:
            continue
        if line[0]==';':
            continue
        dictionary[which].append(line.lower())

def my_tokenizer(text):
    toks=nltk.regexp_tokenize(text,pattern)
    cleanedtoks=[]
    for e in toks:
        word=e.lower()
        if word.isalnum():
            cleanedtoks.append(word)
    return cleanedtoks

def build_feature_matrix(datatype):
    dfname=os.getcwd()+r'\txt_sentoken'+r'\\'+datatype
    r=len(matrix)
    fns=os.listdir(dfname)
    for f in fns:
        if r%500==0:
            print f
        f=open(dfname+r'\\'+f,'r')
        text=f.read().strip()
        tokens=my_tokenizer(text)
        matrix.append([])
        for word in dictionary['negative']:
            if word in tokens:
                matrix[r].append(1)
            else:
                matrix[r].append(0)            
        for word in dictionary['positive']:
            if word in tokens:
                matrix[r].append(1)
            else: matrix[r].append(0)
        labels.append(datatype[0])
        r=r+1

#Classifies according to Naive Bayes and outputs accuracy for given set
#of samples
def naive_bayes_classifier(mle_n,mle_p,samples,labelss,prior_n,prior_p):
    correct=0
    for i in range(samples.shape[0]):
        if i%100==0: print "done "+str(i)
        actual=labelss[i]
        spp=0
        spn=0
        for j in range(mle_n.shape[1]):
            if samples[i,j]==0:
                spp+=math.log(1-mle_p[0,j])
                spn+=math.log(1-mle_n[0,j])
            else:
                spp+=math.log(mle_p[0,j])
                spn+=math.log(mle_n[0,j])
        spp+=math.log(prior_p)
        spn+=math.log(prior_n)
        pred=''
        if spp>spn:
            pred='p'
        else: pred='n'
        if pred==actual:
            correct+=1   

    accuracy=float(correct)/float(samples.shape[0])
    return accuracy

###################################################
#           Main
###################################################
#Implementing 5.1.1
#Read vocab
print "Populating positive words"
populate_list('positive')
print "Populating negative words"
populate_list('negative')

#Implementing 5.1.2
start_time=time.time()
build_feature_matrix('neg')
build_feature_matrix('pos')
end_time=time.time()
print(str((end_time-start_time)/60.0)+" mins.\n")
#matrix built

M=numpy.matrix(matrix)
trainmn=M[0:800,]
trainmp=M[1000:1800,]
trainm=numpy.vstack([M[0:800,],M[1000:1800,]])
testm=numpy.vstack([M[800:1000,],M[1800:,]])
trainlp=labels[0:800]
trainln=labels[1000:1800]
testl=labels[800:1000]+labels[1800:]

mle_n=trainmn.sum(axis=0)
mle_n=mle_n.astype(float)
#use laplace prior of 0.1
mle_n=(mle_n+0.1)/800.2
#checking
mle_na=numpy.asarray(mle_n)
mle_nu=numpy.unique(mle_na)

mle_p=trainmp.sum(axis=0)
mle_p=mle_p.astype(float)
#use laplace prior of 0.1
mle_p=(mle_p+0.1)/800.2
#checking
mle_pa=numpy.asarray(mle_p)
mle_pu=numpy.unique(mle_pa)

#Generating top 10 words in each category
prior_p=0.5
prior_n=0.5
start_time=time.time()
#Implementing 5.2.2
print "Measuring test accuracy"
test_accuracy=naive_bayes_classifier(mle_n,mle_p,testm,testl,prior_n,prior_p)
print "Test Accuracy is: "+str(test_accuracy*100)
#Implementing 5.2.3
print "Measuring train accuracy"
train_accuracy=naive_bayes_classifier(mle_n,mle_p,trainm,trainlp+trainln,prior_n,prior_p)
end_time=time.time()
print(str((end_time-start_time)/60.0)+" mins.\n")
print "Train Accuracy is: "+str(train_accuracy*100)

#Finding top negative words
feature_list=dictionary['negative']+dictionary['positive']
topni=numpy.unique(mle_na,return_index=True)
top10ni=topni[1][-10:]
top10negative=[]
for e in top10ni:
    top10negative.append(feature_list[e])
top10negative=top10negative.reverse()

#Finding top negative words
toppi=numpy.unique(mle_pa,return_index=True)
top10pi=toppi[1][-10:]
top10positive=[]
for e in top10pi:
    top10positive.append(feature_list[e])
top10positive=top10positive.reverse()

