
# coding: utf-8

# # Coursework Part 1: Detecting Spam with Spark
# 
# These are the tasks for IN432 Big Data coursework 2018, part 1.  
# 
# This coursework is about classification of e-mail messages as spam or non-spam in Spark. We will go through the whole process from loading preprocessing to training and testing classifiers in a distributed way in Spark. We wil use the techniques shown in the lextures and labs. I will also introduce here a few additional elements, such as the NLTK and some of the preprocessing and machine learning functions that come with Spark. You are not expected to need anything beyond the material handed out so far and in some cases the Spark documentation, to which I have put links in this document.  
# 
# The structure is similar to the lab sheets. I provide a code structure with gaps that you are supposed to file. In addition you should run 2 small experiments and comment on the results. The lines where you are supposed to add code or take another action are marked with ">>>" 
# please leave the ">>>" in the text, comment out that line, and write your own code in the next line using a copy of that line as a starting point.
# 
# I have added numerous comments in text cells and the code cells to guid you through the program. Please read them carefully and ask if anything is unclear. 
# 
# Once you have completed the tasks, don't delete the outpus, but downlaod the notebook (outputs will be included) and upload it into the coursework submission area on Moodle. The coursework part counts for 25% or the total coursework.

# ## Load and prepare the data
# 
# We will use the lingspam dataset in this coursework (see [http://csmining.org/index.php/ling-spam-datasets.html](http://csmining.org/index.php/ling-spam-datasets.html) for more information).
# 
# The next cell is only needed if you haven't cloned the repository in week 2 or later (but it doesn't do harm to run it). 

# In[2]:


get_ipython().magic(u'cd ~/notebook/work/')
get_ipython().system(u'git clone https://github.com/tweyde/City-Data-Science.git')


# In[3]:


get_ipython().magic(u'cd ~/notebook/work/City-Data-Science/')
get_ipython().system(u'git pull')
get_ipython().magic(u'cd ./datasets/')
#we need to use a magic command (starting with '%') here.
print(">>> Extracting the ling_spam dataset, this can take a moment.")
get_ipython().system(u'tar -xf lingspam_public02.tar.gz')
# '!' calls a program on the machine (the DSX service runs on Linux machines).
print(">>> Unzipping finished.")
# We now have a new dataset in directory 'bare'.
get_ipython().magic(u'cd lingspam_public/bare')
print(">>> pwd ")
get_ipython().system(u'pwd')
print(">>> ls ")
get_ipython().system(u'ls')
# the line before last of output should show "part1 part10 part2  part3  part4  part5  part6  part7 part8 part9"
get_ipython().magic(u'cd ..')


# ### Tools for Troubleshooting
# 
# Normally, DSX works reliably, but there are two issues that have occured. We have solutions for them that you can use with the following cells. 
# If other problems occur, reloading the page and/or restarting the Kernel can help. 

# In[4]:


# try this in case of "sc undefined" or similar errors, should normally not be necessary.
from pyspark import SparkContext
sc = spark.sparkContext


# In[5]:


# sometimes, when you have multiple notebooks open at the same time, you might get an error that the metastore_db is not accessible.
# We can not prevent this form happening on DSX (apart from not opening more than one notebook at a time).
# If it does happen you need to delete the metastore_db. The path of the metastore_db is in the error messages and it's typically 
# lond like this example: 
# /gpfs/global_fs01/sym_shared/YPProdSpark/user/s832-dfe96c6e1f1d61-70d619a53771/notebook/jupyter-rt/kernel-cdcf5f73-9afb-481d-ac40-a210a649eb69-20180222_154448/metastore_db
# once you have it, you can use it with !rm -Rf to delete it:
get_ipython().system(u'rm -Rf <Put the path of the metastore_db here>')


# ## Task 1) Read the dataset and create RDDs 
# a) Start by reading the directory with text files from the file system (`~/notebook/work/City-Data-Science/datasets/bare`). Load all text files per dirctory (part1,part2, ... ,part10) using `wholeTextFiles()`, which creates one RDD per part, containing tuples (filename,text). This is a good choice as the text files are small. (5%)
# 
# b) We will use one of the RDDs as test set, the rest as training set. For the training set you need to create the union of the remaining RDDs. (5%)
# 
# b) Remove the path and extension from the filename using the regular expression provided (5%).
# 
# If the filename starts with 'spmsg' it is spam, otherwise it is not. We'll use that later to train a classifier. 
# 
# We will put the code in each cell into a function that we can reuse later. In this way we can develop the whole preprocessing with the smaller test set and apply it to the training set once we know that everything works. 

# In[6]:


from pathlib import Path
import re

def makeTestTrainRDDs(pathString):
    """ Takes one of the four subdirectories of the lingspam dataset and returns two RDDs one each for testing and training. """
    # We should see10 parts that we can use for creating train and test sets.
    p = Path(pathString) # gets a path object representing the current directory path.
    dirs = list(p.iterdir()) # get the directories part1 ... part10. 
    print(dirs) # Print to check that you have the right directory. You can comment this out when checked. 
    rddList = [] # create a list for the RDDs
    # now create an RDD for each 'part' directory and add them to rddList
    for d in dirs: # iterate through the directories
        rdd = sc.wholeTextFiles(str(d.absolute())) # read the files in the directory 
        rddList.append(rdd) #append the RDD to the rddList
    print('len(rddList)',len(rddList))  # we should now have 10 RDDs in the list # just for testing
    print(rddList[1].take(1)) # just for testing, comment out when it works.
    #this print the first element of our rdd, whose structure is [(filename, text)]. In the tuple, filename is an alphanumeric index qualifying the message contained in the text par, which 
    #in turn contains what appears to be an email message. In the list there are multiple tuples: [(f,t), (f,t), (f,t)...]. 
    
    testRDD1 = rddList[9] # set the test set
    trainRDD1 = rddList[0] # start the training set from 0 and 
    # now loop over the range from 1 to 9(exclusive) to create a union of the remaining RDDs
    for i in range(1,9):
        trainRDD1 = trainRDD1.union(rddList[i]) # create a union of the current and the next 
    # RDD in the list, so that in the end we have a union of all parts 0-8. (9 ist used as test set)
    # both RDDs should remove the paths and extensions from the filename. 
    #>>> This regular expression will do it: re.split('[/\.]', fn_txt[0])[-2]
    #>>> apply it to the filenames in train and test RDD with a lambda
    testRDD2 = testRDD1.map(lambda fn_txt: (re.split('[/\.]', fn_txt[0])[-2], fn_txt[1]))
    trainRDD2 = trainRDD1.map(lambda fn_txt: (re.split('[/\.]', fn_txt[0])[-2], fn_txt[1]))
    return (trainRDD2,testRDD2)

# this makes sure we are in the right directory
get_ipython().magic(u'cd ~/notebook/work/City-Data-Science/datasets/lingspam_public/')
# this should show "bare  lemm  lemm_stop  readme.txt  stop"
get_ipython().system(u'ls')
# the code below is for testing the function makeTestTrainRDDs
trainRDD_testRDD = makeTestTrainRDDs('bare') # read from the 'bare' directory - this takes a bit of time
(trainRDD,testRDD) = trainRDD_testRDD # unpack the returned tuple
print('created the RDDs') # notify the user, so that we can figure out where things went wrong if they do.
print('testRDD.count(): ',testRDD.count()) # 290 
#print('trainRDD.count(): ',trainRDD.count()) # should be 2603 - commented out to save time
print('testRDD.getNumPartitions()',testRDD.getNumPartitions()) # normally 2 on DSX
print('testRDD.getStorageLevel()',testRDD.getStorageLevel()) # Serialized 1x Replicated on DSX
print('testRDD.take(1): ',testRDD.take(30)) # should be (filename,[tokens]) 
rdd1 = testRDD # use this for developemnt in the next tasks 


# ## Task 2) Tokenize and remove punctuation
# 
# Now we need to split the words, a process called *tokenization* by linguists, and remove punctuation. 
# 
# We will use the Python [Natural Language Toolkit](http://www.nltk.org) *NLTK* to do the tokenization (rather than splitting ourselves, as these specialist tools usually do that we can ourselves). We use the NLTK function word_tokenize, see here for a code example: [http://www.nltk.org/book/ch03.html](http://www.nltk.org/book/ch03.html). 5%
# 
# Then we will remove punctuation. There is no specific funtion for this, so we use a regular expression (see here for info [https://docs.python.org/3/library/re.html?highlight=re#module-re](https://docs.python.org/3/library/re.html?highlight=re#module-re)) in a list comprehension (here's a nice visual explanation: [http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/](http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/)). 5% 
# 
# We use a new technique here: we separate keys and values of the RDD, using the RDD functions `keys()` and `values()`, which yield each a new RDD. Then we process the values and *zip* them together with the keys again. See here for documentation: [http://spark.apache.org/docs/2.1.0/api/python/pyspark.html#pyspark.RDD.zip](http://spark.apache.org/docs/2.1.0/api/python/pyspark.html#pyspark.RDD.zip).  We wrap the whole sequence into one function `prepareTokenRDD` for later use. 5%

# In[26]:


import nltk
import re
from nltk.corpus import stopwords

def tokenize(text):
    """ Apply the nltk.word_tokenize() method to our text, return the token list. """
    nltk.download('punkt') # this loads the standard NLTK tokenizer model 
    tokens= nltk.word_tokenize(text)
    # it is important that this is done here in the function, as it needs to be done on every worker.
    # If we do the download outside a this function, it would only be executed on the driver     
    return tokens  # use the nltk function word_tokenize


def removePunctuation(tokens):
    """ Remove punctuation characters from all tokens in a provided list. """
    # this will remove all punctuation from string s: re.sub('[()\[\],.?!";_]','',s)
    tokens2 =  [re.sub('[()\[\],.?!";_]','', x) for x in tokens] # use a list comprehension to remove punctuation
    return tokens2


def prepareTokenRDD(fn_txt_RDD):
    """ Take an RDD with (filename,text) elements and transform it into a (filename,[token ...]) RDD without punctuation characters. """
    rdd_vals2 = fn_txt_RDD.values() # It's convenient to process only the values. 
    rdd_vals3 = rdd_vals2.map(tokenize) # Create a tokenised version of the values by mapping
    rdd_vals4 = rdd_vals3.map(removePunctuation) # remove punctuation from the values
    rdd4 = fn_txt_RDD.keys().zip(rdd_vals4) # we zip the two RDDs together 
    # i.e. produce tuples with one item from each RDD.
    # This works because we have only applied mapping s to the values, 
    # therefore the items in both RDDs are still aligned.
    # now remove any empty value strings (i.e. length 0) that we may have created by removing punctation.
    rdd5 = rdd4.map(lambda x: (x[0], [s for s in x[1] if len(s)>0]))
    rdd6 = rdd5.filter(lambda x: len(x[1])>=0) # do the removal using RDD.filter and a lambda. TIP len(s) gives you the lenght of string. 
    return rdd6

#--------------------------------------------------------------------------------
# Question: why should this be done after zipping the keys and values together?
# Because, as the documentation shows, zip 'assumes that the two RDDs have the same number of partitions and the same number of elements in each partition'. This means in our case that 
# we cannot delete the empty value strings first and then zip the RDDs, because doing so would change the number of elements in the values RDD and thus make the subsequent pairing of key/values impossible, as for some key we would not have 
# the corresponding value. Instead, we first zip the two RDDs, and only afterwards we change the number of elements in the resulting RDD with the deletion of empty  strings. 
# ------------------------------------------------------------------------------

rdd2 = prepareTokenRDD(rdd1) # Use the test set for now, because it is smaller
print(rdd2.take(30)) # For checking result of task 2. 


# ## Task 3) Creating normalised TF.IDF vectors of defined dimensionality, measure the effect of caching.
# 
# We use the hashing trick to create fixed size TF vectors directly from the word list now (slightly different from the previous lab, where we used *(word,count)* pairs.). Write a bit of code as needed. (5%)
# 
# Then we'll use the IDF and Normalizer functions provided by Spark. They use a slightly different pattern than RDD.map and reduce, have a look at the examples here in the documentation for Normalizer  and IDF:
# [http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.feature.Normalizer](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.feature.Normalizer), [http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.feature.IDF](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.feature.IDF) (5%)
# 
# We want control of the dimensionality in the `normTFIDF` function, so we introduce an argument into our functions that enables us to vary dimensionalty later. Here is also an opportunity to benefit from caching, i.e. persisting the RDD after use, so that it will not be recomputed.  (5%)

# In[8]:


# use the hashing trick to create a fixed-size vector from a word list
def hashing_vectorize(text,N): # arguments: the list and the size of the output vector
    v = [0] * N  # create vector of 0s
    for word in text: # iterate through the words 
        h = hash(word)
        v[h % N] = v[h % N] + 1
        # get the hash value 
        # add 1 at the hashed address 
    return v # return hashed word vector

from pyspark.mllib.feature import IDF, Normalizer

def normTFIDF(fn_tokens_RDD, vecDim, caching=True):
    keysRDD = fn_tokens_RDD.keys()
    tokensRDD = fn_tokens_RDD.values()
    tfVecRDD = tokensRDD.map(lambda tokens: hashing_vectorize(tokens,vecDim)) # passing the vecDim value. TIP: you need a lambda. 
    if caching:
        tfVecRDD.persist(StorageLevel.MEMORY_ONLY) # since we will read more than once, caching in Memory will make things quicker.
    idf = IDF() # create IDF object
    idfModel = idf.fit(tfVecRDD) # calculate IDF values
    tfIdfRDD = idfModel.transform(tfVecRDD) # 2nd pass needed (see lecture slides), transforms RDD
    norm = Normalizer() # create a Normalizer object like in the example linked above
    normTfIdfRDD = norm.transform(tfIdfRDD) # and apply it to the tfIdfRDD 
    zippedRDD = keysRDD.zip(normTfIdfRDD) # zip the keys and values together
    return zippedRDD

testDim = 10 # too small for good accuracy, but OK for testing
rdd3 = normTFIDF(rdd2, testDim, True) # test our
print(rdd3.take(1)) # we should now have tuples with ('filename',[N-dim vector])
# e.g. [('9-1142msg1', DenseVector([0.0, 0.0, 0.0, 0.0, 0.4097, 0.0, 0.0, 0.0, 0.9122, 0.0]))]


# ### Task 3a) Caching experiment
# 
# The normTFIDF let's us switch caching on or off. Write a bit of code that measures the effect of caching by takes the time for both options. Use the time function as shown in lecture 3, slide 47. Remember that you need to call an action on an RDD to trigger full execution. 
# 
# Add a short comment on the result (why is there an effect, why of the size that it is?). Remember that this is wall clock time, i.e. you may get noisy results. (10%)

# In[19]:


#run a small experiment with caching set to True or False, 3 times each

from time import time

resCaching = [] # for storing results
resNoCache = [] # for storing results
for i in range(3): # 3 samples
    # start timer
    startTime=time()
    testRDD1 = normTFIDF(rdd2, testDim, True) # 
    testRDD1.count()
    endTime = time()
    resCaching.append(endTime - startTime) # calculate the difference
    
    # start timer
    startTime=time()
    testRDD2 = normTFIDF(rdd2, testDim, False) 
    testRDD2.count()
    endTime = time()
    resNoCache.append(endTime - startTime)
    
import numpy    
meanTimeCaching = numpy.mean(resCaching)
meanTimeNoCache = numpy.mean(resNoCache) # calculate average times

print('Creating TF.IDF vectors, 3 trials - mean time with caching: ', meanTimeCaching, ', mean time without caching: ', meanTimeNoCache)
print('testRDD1.count: ',testRDD1.count())


# Add your results and comments here --------------------------------------------------------------------------------------------

# We decided to compare a very basic action, that is count(), which returns the number of elements stored in rdd2 (291 as can be seen from the last print statement). Both options count the same 291 elements, so the difference
# in time is completely attributable to the presence/absence of caching. Mean time of the caching action was 19.39 seconds; without caching 21.03 seconds. Of course, this is just the result we obtained the last time
# we run this operation. In fact, as as was to be expected, the result varies slighlty each time.  In any case, the action with caching is always ~3 to 5 seconds faster. 
# We have this effect because caching RDDs in Spark allows to speed up applications that access the same RDD multiple times (in our case we run count three times per operation). On the contrary, when the RDD is not 
# cached, it is re-evaluated each time the action count is invoked in the for loop. 
# However, since we chose a very simple action like count, and the operation is iterated only three times, the difference in time is quite small. 


# ## Task 4) Create LabeledPoints 
# 
# Determine whether the file is spam (i.e. the filename contains ’spmsg’) and replace the filename by a 1 (spam) or 0 (non-spam) accordingly. Use `RDD.map()` to create an RDD of LabeledPoint objects. See here [http://spark.apache.org/docs/2.1.0/mllib-linear-methods.html#logistic-regression](http://spark.apache.org/docs/2.1.0/mllib-linear-methods.html#logistic-regression) for an example, and here [http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.regression.LabeledPoint) for the `LabeledPoint` documentation. (10%)
# 
# There is a handy function of Python strings called startswith: e.g. 'abc'.startswith('ab) will return true. The relevant Python syntax here is a conditional expression: **``<a> if <yourCondition> else <b>``**, i.e. 1 if the filename starts with 'spmsg' and otherwise 0.

# In[12]:


from pyspark.mllib.regression import LabeledPoint

# creatate labelled points of vector size N out of an RDD with normalised (filename [(word,count), ...]) items
def makeLabeledPoints(fn_vec_RDD): # RDD and N needed 
    # we determine the true class as encoded in the filename and represent as 1 (samp) or 0 (good)
    cls_vec_RDD = fn_vec_RDD.map(lambda x: (1, x[1]) if x[0].startswith('spmsg') else (0, x[1])) # use a conditional expression to get the class label (True or False)
    # now we can create the LabeledPoint objects with (class,vector) arguments
    lp_RDD = cls_vec_RDD.map(lambda cls_vec: LabeledPoint(cls_vec[0],cls_vec[1]) ) 
    return lp_RDD 

# for testing
testLpRDD = makeLabeledPoints(rdd3) 
print(testLpRDD.take(1)) 
# should look similar to this: [LabeledPoint(0.0, [0.0,0.0,0.0,0.0,0.40968062880166006,0.0,0.0,0.0,0.9122290186048,0.0])]


# ## Task 5) Complete the preprocessing 
# 
# It will be useful to have a single function to do the preprocessing. So integrate everything here. (5%)

# In[13]:


# now we can apply the preprocessing chain to the data loaded in task 1 
# N is for controlling the vector size
def preprocess(rawRDD,N):
    """ take a (filename,text) RDD and transform into LabelledPoint objects 
        with class labels and a TF.IDF vector with N dimensions. 
    """
    tokenRDD = prepareTokenRDD(rawRDD) # task 2
    tfIdfRDD = normTFIDF(tokenRDD,N) # task 3
    lpRDD = makeLabeledPoints(tfIdfRDD) # task 4
    return lpRDD # return RDD with LabeledPoints

# and with this we can start the whole process from a directory, N is again the vector size
def loadAndPreprocess(directory,N):
    """ load lingspam data from a directory and create a training and test set of preprocessed data """
    trainRDD_testRDD = makeTestTrainRDDs('bare') # read from the 'bare' directory - this takes a bit of time
    (trainRDD,testRDD) = trainRDD_testRDD
    return (preprocess(trainRDD,N),preprocess(testRDD,N)) # apply the preprocessing funcion defined above

trainLpRDD = preprocess(trainRDD,testDim) # prepare the training data
print(testLpRDD.take(1)) # should look similar to previous cell's output

train_test_LpRDD = loadAndPreprocess('lemm',100) # let's re-run with another vector size
(trainLpRDD,testLpRDD) = train_test_LpRDD 
print(testLpRDD.take(1))
print(trainLpRDD.take(1))


# ## Task 6) Train some classifiers 
# 
# Use the `LabeledPoint` objects to train a classifier, specifically the *LogisticRegression*, *Naive Bayes*, and *Support Vector Machine*. Calculate the accuracy of the model on the training set (again, follow this example [http://spark.apache.org/docs/2.1.0/ml-classification-regression.html#logistic-regression](http://spark.apache.org/docs/2.0.0/ml-classification-regression.html#logistic-regression) and here is the documentation for the classifiers [LogisticRegressionWithLBFGS](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.LogisticRegressionWithLBFGS), [NaiveBayes](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.NaiveBayes), [SVMWithSGD](http://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.classification.SVMWithSGD).  (10%) 

# In[16]:


from pyspark.mllib.classification import (NaiveBayes, LogisticRegressionWithLBFGS, SVMWithSGD) 
import numpy

# train the model with an (f,[(w,c), ...]) RDD. This is practical as we can reuse the function for TF.IDF
def trainModel(lpRDD):
    """ Train 3 classifier models on the given RDD with LabeledPoint objects. A list of trained model is returned. """
    lpRDD.persist(StorageLevel.MEMORY_ONLY) # not really needed as the Spark implementations ensure caching themselves. Other implementations might not, however. 
    # Train a classifier model.
    print('Starting to train the model') #give some immediate feedback
    model1 = LogisticRegressionWithLBFGS.train(lpRDD) # this is the best model
    print('Trained LR (model1)')
    #print(type(model1))
    model2 = NaiveBayes.train(lpRDD) # doesn't work well
    print('Trained NB (model2)')
    print(type(model2))
    model3 = SVMWithSGD.train(lpRDD) # or this ...
    print('Trained SVM (model3)')
    return [model1,model2,model3]

def testModel(model, lpRDD):
    """ Tests the classification accuracy of the given model on the given RDD with LabeledPoint objects. """
    lpRDD.persist(StorageLevel.MEMORY_ONLY)
    # Make prediction and evaluate training set accuracy.
    # Get the prediction and the ground truth label
    predictionAndLabel = lpRDD.map(lambda p: (model.predict(p.features), p.label)) # get the prediction and ground truth (label) for each item. (0, 1)
    correct = predictionAndLabel.filter(lambda xv: xv[0] == xv[1]).count() # count the correct predictions 25
    accuracy =  1.0*correct/lpRDD.count() # and calculate the accuracy lpRDD.count()/correct
    print('Accuracy {:.1%} (data items: {}, correct: {})'.format(accuracy,lpRDD.count(), correct)) # report to console
    return accuracy # and return the value  

models = trainModel(trainLpRDD) # just for testing
testModel(models[2], trainLpRDD) # just for testing


# In[15]:


get_ipython().system(u'rm -Rf /gpfs/global_fs01/sym_shared/YPProdSpark/user/sb58-fd12fb10398921-2dd0a6f275af/notebook/jupyter-rt/kernel-4ce3b9e2-d9ac-4512-a93a-c7a2636eef04-20180305_071320/metastore_db')


# ## Task 7) Automate training and testing
# 
# We automate now the whole process from reading the files, through preprocessing, and training up to evaluating the models. In the end we have a single function that takes all the parameters we are interested in and produces trained models and an evaluation. (5%) 

# In[17]:


# this function combines tasks f) and g)
# this method should take RDDs with (f,[(w,c), ...])
def trainTestModel(trainRDD,testRDD):
    """ Trains 3 models and tests them on training and test data. Returns a matrix the training and testing (rows) accuracy values for all models (columns). """
    models = [LogisticRegressionWithLBFGS, NaiveBayes, SVMWithSGD] # train models on the training set
    results = [[], []] # matrix for 2 modes (training/test) vs n models (currently 3)
    for mdl in models:
        print('Training')
        model= mdl.train(trainRDD)
        results[0].append(testModel(model, trainRDD)) # test the model on the training set
        print('Testing')
        results[1].append(testModel(model, testRDD)) # test the model on the test set
    return results

def trainTestFolder(folder,N):
    """ Reads data from a folder, preproceses the data, and trains and evaluates models on it. """
    print('Start loading and preprocessing') 
    train_test_LpRDD = loadAndPreprocess(folder,N) # create the RDDs
    print('Finished loading and preprocessing')
    (trainLpRDD,testLpRDD) = train_test_LpRDD # unpack the RDDs 
    return trainTestModel(trainLpRDD,testLpRDD) # train and test

trainTestFolder('lemm',1000) 


# ## Task 8) Run experiments 
# 
# We have now a single function that allows us to vary the vector size easily. Test vector sizes 5, 50, 500, 5000, 50000 and examine the effect on the classification accuracy in Experiment 1.
# 
# Use the function from Task 7) to test different data types. The dataset has raw text in folder `bare`, lemmatised text in  `lemm` (similar to stemming, reduces to basic word forms), `stop` (with stopwords removed), and `lemm_stop` (lemmatised and stopwords removed). Test how the classification accuracy differs for these four data types in Experiment 2. Collect the results in a data structure that can be saved for later saving and analyis.
# 
# Comment on the results in a few sentences, considering the differences in performance between the different conditions as well as train an test values. 15%

# In[18]:


from pyspark.sql import DataFrame

folder = 'bare'
N = numpy.array([3,30,300,3000,30000]) 
print('\nEXPERIMENT 1: Testing different vector sizes')
results = []
for n in N:
    print('N = {}'.format(n))
    results.append(trainTestFolder(folder,n))
    
n = 3000
typeFolders = ['bare','stop','lemm','lemm_stop']
print('EXPERIMENT 2: Testing different data types')
for tpf in typeFolders:
    print('Path = {}'.format(tpf))
    results.append(trainTestFolder(tpf,n))

# Add comments on the performance in a cell below. 


# In[ ]:


# Comments -----------------------------------------------------------------------------------------

# 1) EXPERIMENT 1 - different vector sizes
# With N=3 the three methods (Logistic Regression, NaiveBayes, and SVM) have the same performances, ~ 83%. Already at N= 30 they start to diverge. In fact, Logistic regression 
# accuracy increases both in train and test (respectively +2% and +3%), while the other models remain stable. This pattern gets confirmed at N=300, where Logistic Regression reach its best possible performance 
# for training data (100%). At this stage also the training accuracy of Naive Bayes starts improving (+3% training accuracy). At N=3000 Naive Bayes reaches its best performances in both training (97.5%) and testing data (95.9%). 
# These percentages suddenly worsen at N=30000, where NB drops to 86.1% (train) - 84.5% (test). On the other hand, Logistic regression keeps improving, reaching a testing accuracy of 98.3%.
# In conclusion, if we increase the number of features used for the spam/ham classification:  
# 1.a the performance of SVM does not vary, and remains the worse model for classification;
# 1.b Logistic regression steadily improves with the addition of features (N) for the classification and may reach up to 98% of testing accuracy at N=30000;
# 1.c Naive Bayes performance improves up till N=3000 features, coming very close to logistic regression accuracy (-2%). 
# However, increasing the number of features to N=30000 will only decrease NB accuracy. We think this behavior of NB may be linked to its assumption of independency. Given a certain train/test sample, NB can use only a certain number of 
# features. If we continue adding features, they may start to get redundant and correlated, which is directly against the independence assumption - hence the drop in performance. In this scenario Logistic regression is better 
# suited for classification, because the addition of features can only improve its performance - there is no independence assumption to deal with.   

# 2) EXPERIMENT 2 - different data types
# With NB and N=3000 we obtain the same results regardless of the data type tested. This means that further trimming (indeed we already deleted all punctuation)
# words with stopwords and/or stemming does not add any predictive power to our classifiers.  

