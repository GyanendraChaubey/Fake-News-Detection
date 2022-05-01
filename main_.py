#Import libraries
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#Import data set
data=pd.read_csv('D:/Downloads/news_articles.csv')
data.info()

data.isnull().sum()
data_1=data.dropna()
data_1.head()

#data Visualization
#plot number of real and fake news count
counts = data_1['label'].value_counts()
data_1.label.value_counts()
plot=sns.countplot(x='label',data=data_1);

# Plot article type distribution
data_1_type = data_1['type'].value_counts()
sns.barplot(np.arange(len(data_1_type)), data_1_type)
plt.xticks(np.arange(len(data_1_type)), data_1_type.index.values.tolist())
plt.title('Article type count', fontsize=20)
plt.show()

#plot article type count distribution
plt.figure(figsize = (7,7))
type_counts = data_1['type'].value_counts()
plt.pie(type_counts, labels = type_counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Total counts: {}'.format(data_1.shape[0]))
plt.title('Type count', fontsize = 15);

#Text Preprocessing and Bag of Words 
import nltk
nltk.download('stopwords')

stopwords.words('english')[0:10] 

def text_process(mess):
    no_punctuation = [char for char in mess if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
    
    lemmatizer = nlp.WordNetLemmatizer()
    no_punctuation = [ lemmatizer.lemmatize(word) for word in no_punctuation]
    data_1['title'].apply(text_process)
    data_1['text'].head(5).apply(text_process)
    data_1['text'].head(5).apply(text_process)
    
#CountVectorizer method
bow_trans = CountVectorizer(analyzer=text_process).fit(data_1['text'])

print(len(bow_trans.vocabulary_))
msg_bow = bow_trans.transform(data_1['text'])

tfidf_transformer = TfidfTransformer().fit(msg_bow)

msg_tfidf = tfidf_transformer.transform(msg_bow)
print(msg_tfidf.shape)

#label encoding
le = LabelEncoder()
y = le.fit_transform(data_1.label)

#Splitting the data set into the Training set and Test 
X_train, X_test, y_train, y_test = train_test_split(data_1['text'], y, test_size=0.2, random_state = 42)

#Classification
#Decision Trees
dt = DecisionTreeClassifier(random_state=42, criterion="entropy",min_samples_split=10, min_samples_leaf=10, max_depth=3, max_leaf_nodes=5)
pipeline_dt = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)), 
    ('tfidf', TfidfTransformer()),  
    ('classifier', DecisionTreeClassifier()),  
])
pipeline_dt.fit(X_train,y_train)
pred_dt = pipeline_dt.predict(X_test)


#Analysis Report
print("PERFORMANCE ANALYSIS")
print("-------------------")
print("Decision Tree")
print("------Classification Report------")
print(classification_report(pred_dt,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{round(accuracy_score(pred_dt,y_test)*100)}")


#Gradient Boosting
gbm=GradientBoostingClassifier(learning_rate=0.3,max_depth=4,n_estimators=100 ,random_state=0)

pipeline_gb = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', GradientBoostingClassifier()),  
])
pipeline_gb.fit(X_train,y_train)
pred_gb = pipeline_gb.predict(X_test)

#Analysis Report
print("-------------------")
print("Gradient Boosting")
print("------Classification Report------")
print(classification_report(pred_gb,y_test))

print()
print("------Accuracy------")
print(f"The Accuracy Score :{round(accuracy_score(pred_gb,y_test)*100)}")


