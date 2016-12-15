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
adverbs=["extremely","quite","just","almost","very","too","enough"]
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
        word=line.lower()
        dictionary[which].append(word)
        for adverb in adverbs:
            dictionary[which].append(adverb+'_'+word)

def my_tokenizer(text):
    toks=nltk.regexp_tokenize(text,pattern)
    cleanedtoks=[]
    for e in toks:
        word=e.lower()
#if word not in stopwords:
        if word.isalnum():
            cleanedtoks.append(word)
    return cleanedtoks

def build_feature_matrix(datatype):
    dfname=os.getcwd()+r'\txt_sentoken'+r'\\'+datatype
    nf=len(dictionary['positive'])+len(dictionary['negative'])
    r=len(matrix)
    fns=os.listdir(dfname)
    for f in fns:
        if r%500==0:
            print f
        f=open(dfname+r'\\'+f,'r')
        text=f.read().strip()
        tokens=my_tokenizer(text)
        matrix.append([])
        matrix[r]=[0]*nf
        for i in range(len(tokens)):
            if tokens[i] in dictionary['negative']:
                fi=dictionary['negative'].index(tokens[i])
                matrix[r][fi]=1
                if (i-1)>0:
                    if tokens[i-1] in adverbs:
                        fi=dictionary['negative'].index(tokens[i-1]+'_'+tokens[i])
                        matrix[r][fi]=1
            if tokens[i] in dictionary['positive']:
                fi=dictionary['positive'].index(tokens[i])
                matrix[r][fi]=1
                if (i-1)>0:
                    if tokens[i-1] in adverbs:
                        fi=dictionary['positive'].index(tokens[i-1]+'_'+tokens[i])
                        matrix[r][fi]=1
                
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

print "Populating positive words"
populate_list('positive')
print "Populating negative words"
populate_list('negative')
dictionary['not']=[]
###################################################
#           Main
###################################################

#vocab built
start_time=time.time()
build_feature_matrix('neg')
build_feature_matrix('pos')
end_time=time.time()
print(str((end_time-start_time)/60.0)+" mins.\n")
#matrix built
M=numpy.matrix(matrix)
print M.shape
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

prior_p=0.5
prior_n=0.5
start_time=time.time()
print "Measuring test accuracy"
test_accuracy=measure_accuracy(mle_n,mle_p,testm,testl,prior_n,prior_p)
print "Test Accuracy is: "+str(test_accuracy*100)
print "Measuring train accuracy"
train_accuracy=measure_accuracy(mle_n,mle_p,trainm,trainlp+trainln,prior_n,prior_p)
end_time=time.time()
print(str((end_time-start_time)/60.0)+" mins.\n")
print "Train Accuracy is: "+str(train_accuracy*100)

