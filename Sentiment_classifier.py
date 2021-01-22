#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

class Sentiment:
    NEGATIVE="NEGATIVE"
    NEUTRAL="NEUTRAL"
    POSITIVE="POSITIVE"
class Review:
    def __init__(self,text,score):
        self.text=text
        self.score=score
        self.sentiment=self.get_sentiment()
    def get_sentiment(self):
        if self.score <=2:
            return Sentiment.NEGATIVE
        elif self.score ==3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE
        
class ReviewContainer:
    def __init__(self,reviews):
        self.reviews=reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative=list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE,self.reviews))
        positive=list(filter(lambda x: x.sentiment == Sentiment.POSITIVE,self.reviews))
        positive_shrunk=positive[:len(negative)]
        self.reviews=negative+positive_shrunk
        random.shuffle(self.reviews)
        


# In[2]:


import json

file_name="Books_small_10000.json"

reviews=[]
with open(file_name) as f:
    for line in f:
        review=json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))
reviews[4].text


# In[3]:


from sklearn.model_selection import train_test_split
training,test=train_test_split(reviews,test_size=0.33,random_state=42)

train_container=ReviewContainer(training)

test_container=ReviewContainer(test)


# In[14]:


train_container.evenly_distribute()
train_x=train_container.get_text()
train_y=train_container.get_sentiment()

test_container.evenly_distribute()
test_x=test_container.get_text()
test_y=test_container.get_sentiment()

print(train_y.count(Sentiment.NEGATIVE))
print(train_y.count(Sentiment.POSITIVE))
      


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors=vectorizer.fit_transform(train_x)

test_x_vectors=vectorizer.transform(test_x)

print(train_x[0])
print(train_x_vectors[0].toarray())


# #### SVM Classifier

# #### Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. But generally, they are used in classification problems.An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH).
# #### The followings are important concepts in SVM −
# #### 1) Support Vectors − Datapoints that are closest to the hyperplane is called support vectors. Separating line will be defined with the help of these data points.
# 
# #### 2) Hyperplane − As we can see in the above diagram, it is a decision plane or space which is divided between a set of objects having different classes.
# 
# #### 3) Margin − It may be defined as the gap between two lines on the closet data points of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. Large margin is considered as a good margin and small margin is considered as a bad margin.
# 

# In[6]:


from sklearn import svm

clf_svm=svm.SVC(kernel='linear')

clf_svm.fit(train_x_vectors,train_y)

test_x[0]

clf_svm.predict(test_x_vectors[0])


# #### Decision Tree Classifier

# #### A Decision Tree is a simple representation for classifying examples. It is a Supervised Machine Learning where the data is continuously split according to a certain parameter.
# #### Decision Tree consists of :
# #### 1) Nodes : Test for the value of a certain attribute.
# #### 2) Edges/ Branch : Correspond to the outcome of a test and connect to the next node or leaf.
# #### 3) Leaf nodes : Terminal nodes that predict the outcome (represent class labels or class distribution).
# #### 1. Classification trees (Yes/No types) :
# #### What we’ve seen above is an example of classification tree, where the outcome was a variable like ‘fit’ or ‘unfit’. Here the decision variable is Categorical/ discrete.
# #### Such a tree is built through a process known as binary recursive partitioning. This is an iterative process of splitting the data into partitions, and then splitting it up further on each of the branches.
# #### 2. Regression trees (Continuous data types) :
# #### Decisionwhere the target variable can take continuous values (typically real numbers) are called regression trees. (e.g. the price of a house, or a patient’s length of stay in a hospital)

# In[7]:


from sklearn.tree import DecisionTreeClassifier

clf_dec=DecisionTreeClassifier()
clf_dec.fit(train_x_vectors,train_y)

clf_dec.predict(test_x_vectors[0])


# #### Logistic Regression

# #### Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.
# #### For example,
# #### 1)To predict whether an email is spam (1) or (0)
# #### 2)Whether the tumor is malignant (1) or not (0)
# #### Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.From this example, it can be inferred that linear regression is not suitable for classification problem. Linear regression is unbounded, and this brings logistic regression into picture. Their value strictly ranges from 0 to 1.

# In[8]:


from sklearn.linear_model import LogisticRegression

clf_log=LogisticRegression()
clf_log.fit(train_x_vectors,train_y)

clf_dec.predict(test_x_vectors[0])


# #### Mean Accuracy

# In[9]:


print(clf_svm.score(test_x_vectors,test_y))
print(clf_log.score(test_x_vectors,test_y))
print(clf_dec.score(test_x_vectors,test_y))


# #### F1 Score

# In[10]:


from sklearn.metrics import f1_score

f1_score(test_y,clf_svm.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEGATIVE])
#f1_score(test_y,clf_dec.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])
#f1_score(test_y,clf_log.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])


# #### Output

# In[13]:


test_set=["Please dont ever buy this product guyzz....its my request....look at this picture....the pencils would look good as hell at first sight....but when u sharpen them...then u will see how worst a pencil could ever be.... please dont buy it....dont waste your money.... ","Don't buy this. Everytime you try to sharpen it, the needle keeps on breaking.. Pencils are very weak.. ","Super!!! Loved it!","Not worth your money","average"]
new_test=vectorizer.transform(test_set)

clf_svm.predict(new_test)


# In[ ]:





# In[ ]:





# In[ ]:




