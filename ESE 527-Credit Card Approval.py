#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# In[2]:


application = pd.read_csv("application_record.csv", encoding = 'utf-8') 
credit = pd.read_csv("credit_record.csv", encoding = 'utf-8')  


# In[3]:


#plt.rcParams['figure.facecolor'] = 'white'


# # Brief Analysis of Binary and Continuous Features

# ### Find out the percentage of approved and rejected clients - using yes and no

# In[4]:


# find all users' account open month.
begin_month=pd.DataFrame(credit.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'begin_month'}) 
new_data=pd.merge(application,begin_month,how="left",on="ID") #merge to record data


# In[5]:


credit['dep_value'] = None
credit['dep_value'][credit['STATUS'] =='2']='Yes' 
credit['dep_value'][credit['STATUS'] =='3']='Yes' 
credit['dep_value'][credit['STATUS'] =='4']='Yes' 
credit['dep_value'][credit['STATUS'] =='5']='Yes' 


# In[6]:


cpunt=credit.groupby('ID').count()
cpunt['dep_value'][cpunt['dep_value'] > 0]='Yes' 
cpunt['dep_value'][cpunt['dep_value'] == 0]='No' 
cpunt = cpunt[['dep_value']]
new_data=pd.merge(new_data,cpunt,how='inner',on='ID')
new_data['target']=new_data['dep_value']
new_data.loc[new_data['target']=='Yes','target']=1
new_data.loc[new_data['target']=='No','target']=0


# In[7]:


print(cpunt['dep_value'].value_counts())
cpunt['dep_value'].value_counts(normalize=True)


# ### Rename all features

# In[8]:


# RENAME FEATURES

new_data.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car','FLAG_OWN_REALTY':'Reality',
                         'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'inc',
                         'NAME_EDUCATION_TYPE':'edutp','NAME_FAMILY_STATUS':'famtp',
                        'NAME_HOUSING_TYPE':'houtp','FLAG_EMAIL':'email',
                         'NAME_INCOME_TYPE':'inctp','FLAG_WORK_PHONE':'wkphone',
                         'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',
                        'OCCUPATION_TYPE':'occyp'
                        },inplace=True)


# In[9]:


new_data.dropna()
new_data = new_data.mask(new_data == 'NULL').dropna()


# In[10]:


ivtable=pd.DataFrame(new_data.columns,columns=['variable'])
ivtable['IV']=None
namelist = ['FLAG_MOBIL','begin_month','dep_value','target','ID']

for i in namelist:
    ivtable.drop(ivtable[ivtable['variable'] == i].index, inplace=True)


# ### Functions of IV and WoE

# In[11]:


# Calculate information value
def calc_iv(df, feature, target, pr=False):
    lst = []
    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    print('This variable\'s IV is:',iv)
    print(df[feature].value_counts())
    return iv, data


# In[12]:


def convert_dummy(df, feature,rank=0):
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df


# In[13]:


def get_category(df, col, binsnum, labels, qcut = False):
    if qcut:
        localdf = pd.qcut(df[col], q = binsnum, labels = labels) # quantile cut
    else:
        localdf = pd.cut(df[col], bins = binsnum, labels = labels) # equal-length cut
        
    localdf = pd.DataFrame(localdf)
    name = 'gp' + '_' + col
    localdf[name] = localdf[col]
    df = df.join(localdf[name])
    df[name] = df[name].astype(object)
    return df


# In[14]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Brief analysis of all features

# #### Binary Features

# ###### Gender

# In[15]:


new_data['Gender'] = new_data['Gender'].replace(['F','M'],[0,1])
print(new_data['Gender'].value_counts())
iv, data = calc_iv(new_data,'Gender','target')
ivtable.loc[ivtable['variable']=='Gender','IV']=iv
data.head()


# ###### Having a car or not

# In[16]:


new_data['Car'] = new_data['Car'].replace(['N','Y'],[0,1])
print(new_data['Car'].value_counts())
iv, data=calc_iv(new_data,'Car','target')
ivtable.loc[ivtable['variable']=='Car','IV']=iv
data.head()


# ###### Having house reality or not

# In[17]:


new_data['Reality'] = new_data['Reality'].replace(['N','Y'],[0,1])
print(new_data['Reality'].value_counts())
iv, data=calc_iv(new_data,'Reality','target')
ivtable.loc[ivtable['variable']=='Reality','IV']=iv
data.head()


# ###### Having a phone or not

# In[18]:


new_data['phone']=new_data['phone'].astype(str)
print(new_data['phone'].value_counts(normalize=True,sort=False))
new_data.drop(new_data[new_data['phone'] == 'nan' ].index, inplace=True)
iv, data=calc_iv(new_data,'phone','target')
ivtable.loc[ivtable['variable']=='phone','IV']=iv
data.head()


# ###### Having an email or not

# In[19]:


print(new_data['email'].value_counts(normalize=True,sort=False))
new_data['email']=new_data['email'].astype(str)
iv, data=calc_iv(new_data,'email','target')
ivtable.loc[ivtable['variable']=='email','IV']=iv
data.head()


# ###### Having a Work phone or not

# In[20]:


new_data['wkphone']=new_data['wkphone'].astype(str)
iv, data = calc_iv(new_data,'wkphone','target')
new_data.drop(new_data[new_data['wkphone'] == 'nan' ].index, inplace=True)
ivtable.loc[ivtable['variable']=='wkphone','IV']=iv
data.head()


# #### Continuous Features

# ###### Children Numbers

# In[21]:


new_data.loc[new_data['ChldNo'] >= 2,'ChldNo']='2More'
print("total number of values: ", new_data['ChldNo'].value_counts(sort=False))

iv, data=calc_iv(new_data,'ChldNo','target')
ivtable.loc[ivtable['variable']=='ChldNo','IV']=iv
data.head()


# In[22]:


new_data = convert_dummy(new_data,'ChldNo')


# ###### Annual Income

# In[23]:


new_data['inc']=new_data['inc'].astype(object)
new_data['inc'] = new_data['inc']/10000 
print(new_data['inc'].value_counts(bins=10,sort=False))
new_data['inc'].plot(kind='hist',bins=50,density=True)


# In[24]:


new_data = get_category(new_data,'inc', 3, ["low","medium", "high"], qcut = True)
iv, data = calc_iv(new_data,'gp_inc','target')
ivtable.loc[ivtable['variable']=='inc','IV']=iv
data.head()

new_data = convert_dummy(new_data,'gp_inc')


# ###### Age

# In[25]:


new_data['Age']=-(new_data['DAYS_BIRTH'])//365	
print(new_data['Age'].value_counts(bins=10,normalize=True,sort=False))
new_data['Age'].plot(kind='hist',bins=20,density=True)


# In[26]:


new_data = get_category(new_data,'Age',5, ["lowest","low","medium","high","highest"])
iv, data = calc_iv(new_data,'gp_Age','target')
ivtable.loc[ivtable['variable']=='DAYS_BIRTH','IV'] = iv
data.head()

new_data = convert_dummy(new_data,'gp_Age')


# ###### Working Years

# In[27]:


new_data['worktm']=-(new_data['DAYS_EMPLOYED'])//365	
new_data[new_data['worktm']<0] = np.nan # replace by na
new_data['DAYS_EMPLOYED']
new_data['worktm'].fillna(new_data['worktm'].mean(),inplace=True) #replace na by mean
new_data['worktm'].plot(kind='hist',bins=20,density=True)


# In[28]:


new_data = get_category(new_data,'worktm',5, ["lowest","low","medium","high","highest"])
iv, data=calc_iv(new_data,'gp_worktm','target')
ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','IV']=iv
data.head()

new_data = convert_dummy(new_data,'gp_worktm')


# ###### Family size

# In[29]:


new_data['famsize'].value_counts(sort=False)


# In[30]:


new_data['famsize']=new_data['famsize'].astype(int)
new_data['famsizegp']=new_data['famsize']
new_data['famsizegp']=new_data['famsizegp'].astype(object)
new_data.loc[new_data['famsizegp']>=3,'famsizegp']='3more'
iv, data=calc_iv(new_data,'famsizegp','target')
ivtable.loc[ivtable['variable']=='famsize','IV']=iv
data.head()


# In[31]:


new_data = convert_dummy(new_data,'famsizegp')


# #### Categorical Features

# ###### Income type

# In[32]:


print(new_data['inctp'].value_counts(sort=False))
print(new_data['inctp'].value_counts(normalize=True,sort=False))
new_data.loc[new_data['inctp']=='Pensioner','inctp']='State servant'
new_data.loc[new_data['inctp']=='Student','inctp']='State servant'
iv, data=calc_iv(new_data,'inctp','target')
ivtable.loc[ivtable['variable']=='inctp','IV']=iv
data.head()


# In[33]:


new_data = convert_dummy(new_data,'inctp')


# ###### Occupation type

# In[34]:


new_data.loc[(new_data['occyp']=='Cleaning staff') | (new_data['occyp']=='Cooking staff') | (new_data['occyp']=='Drivers') | (new_data['occyp']=='Laborers') | (new_data['occyp']=='Low-skill Laborers') | (new_data['occyp']=='Security staff') | (new_data['occyp']=='Waiters/barmen staff'),'occyp']='Laborwk'
new_data.loc[(new_data['occyp']=='Accountants') | (new_data['occyp']=='Core staff') | (new_data['occyp']=='HR staff') | (new_data['occyp']=='Medicine staff') | (new_data['occyp']=='Private service staff') | (new_data['occyp']=='Realty agents') | (new_data['occyp']=='Sales staff') | (new_data['occyp']=='Secretaries'),'occyp']='officewk'
new_data.loc[(new_data['occyp']=='Managers') | (new_data['occyp']=='High skill tech staff') | (new_data['occyp']=='IT staff'),'occyp']='hightecwk'
print(new_data['occyp'].value_counts())
iv, data=calc_iv(new_data,'occyp','target')
ivtable.loc[ivtable['variable']=='occyp','IV']=iv
data.head()         


# In[35]:


new_data = convert_dummy(new_data,'occyp')


# ###### House type

# In[36]:


iv, data=calc_iv(new_data,'houtp','target')
ivtable.loc[ivtable['variable']=='houtp','IV']=iv
data.head()


# In[37]:


new_data = convert_dummy(new_data,'houtp')


# ###### Education

# In[38]:


new_data.loc[new_data['edutp']=='Academic degree','edutp']='Higher education'
iv, data=calc_iv(new_data,'edutp','target')
ivtable.loc[ivtable['variable']=='edutp','IV']=iv
data.head()


# In[39]:


new_data = convert_dummy(new_data,'edutp')


# ###### Marriage condition

# In[40]:


new_data['famtp'].value_counts(normalize=True,sort=False)


# In[41]:


iv, data=calc_iv(new_data,'famtp','target')
ivtable.loc[ivtable['variable']=='famtp','IV']=iv
data.head()


# In[42]:


new_data = convert_dummy(new_data,'famtp')


# #### IV & WOE of features

# In[43]:


ivtable=ivtable.sort_values(by='IV',ascending=False)
ivtable.loc[ivtable['variable']=='DAYS_BIRTH','variable']='agegp'
ivtable.loc[ivtable['variable']=='DAYS_EMPLOYED','variable']='worktmgp'
ivtable.loc[ivtable['variable']=='inc','variable']='incgp'
ivtable


# # Model fitting and performance

# ### Reconstruct the features

# In[44]:


new_data.columns


# In[45]:


new_data


# In[46]:


Y = new_data['target']
X = new_data[['Gender','Reality','ChldNo_1', 'ChldNo_2More','wkphone',
              'gp_Age_high', 'gp_Age_highest', 'gp_Age_low',
       'gp_Age_lowest','gp_worktm_high', 'gp_worktm_highest',
       'gp_worktm_low', 'gp_worktm_medium','occyp_hightecwk', 
              'occyp_officewk','famsizegp_1', 'famsizegp_3more',
       'houtp_Co-op apartment', 'houtp_Municipal apartment',
       'houtp_Office apartment', 'houtp_Rented apartment',
       'houtp_With parents','edutp_Higher education',
       'edutp_Incomplete higher', 'edutp_Lower secondary','famtp_Civil marriage',
       'famtp_Separated','famtp_Single / not married','famtp_Widow']]


# ### Use SMOTE to balance the data

# In[47]:


from imblearn.over_sampling import SMOTE


# In[48]:


Y = Y.astype('int')
X_balance, Y_balance = SMOTE().fit_resample(X,Y)
X_balance = pd.DataFrame(X_balance, columns = X.columns)


# ### Split the balanced data set

# In[49]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_balance, Y_balance, stratify=Y_balance, test_size=0.3, random_state = 10086)

# set random seed
np.random.seed(100)


# ### Logistic regression

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# In[51]:


for i in range(1, 6):
    model = LogisticRegression(C=i, random_state=0, solver='lbfgs')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# Here we apply the method of grid search for each hyperparameter we want to select.
# 
# As we can see, there is no much difference between the accuracy when we change c, which is the inverse of regularization strength.
# 
# So then we try to change the solver with c fixed as 0.8.

# In[52]:


for i in ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'):
    model = LogisticRegression(C=0.8, random_state=0, solver=i)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# Still, there is no much big difference. After comparing the results, we decide to use c as 0.8 and the solver as lbfgs.

# In[53]:


model_lr = LogisticRegression(C=0.8, random_state=0, solver='lbfgs')
fit_lr = model.fit(X_train, y_train)
y_predict_lr = model.predict(X_test)

# print the accuracy and the confusion matrix
print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))

# plot the confusion matrix
class_names = ['0','1']
plot_confusion_matrix(confusion_matrix(y_test,y_predict), classes= class_names, normalize = True, 
                      title='Normalized Confusion Matrix: Logistic Regression')


# ### Cross validation of logistic regression

# In[65]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_lr, X_test, y_test, cv=5, scoring='f1')

print("Logistic regression: " + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# ### Random forest

# In[55]:


np.random.seed(100)
for esti in (50, 100, 150, 200, 250, 300):
    model = RandomForestClassifier(n_estimators=esti, max_depth=12, min_samples_leaf=16)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# As we can see, the best accuracy is reached when the number of estimators is 100.(Although the result may fluctuate a little.)

# In[56]:


for i in (1,5,10,15,20,25,30,40):
    model = RandomForestClassifier(n_estimators=100, max_depth=i, min_samples_leaf=16)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# We could find that when max depth increases, the accuracy will also increase. 
# 
# But after max depth reaches 20, the accuracy fluctuate near 0.87. So here we use max_depth=20.

# In[57]:


for i in (1,5,10,15,16,20):
    model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=i)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# There is no big difference between the results when we change the min samples leaf. Here we use min_samples_leaf=16.

# In[58]:


model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=16)
fit_rf = model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))
print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
plot_confusion_matrix(confusion_matrix(y_test,y_predict), classes=class_names, normalize = True, 
                      title='Normalized Confusion Matrix: Ramdom Forests')


# ### Cross validation of random forest

# In[59]:


scores = cross_val_score(model_rf, X_test, y_test, cv=5, scoring='f1')
print("Random forest: " + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# ### Support vector machine (SVM)

# In[60]:


from sklearn import svm
for i in (0.05, 0.1, 1, 5, 10):
    model = svm.SVC(C = i, kernel='linear')
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# The accuracy does not change much when the value of C is bigger than 0.1. So here we just use C=0.8 and further 

# In[61]:


for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
    model = svm.SVC(C = 0.8, kernel=kernel)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))


# As we can see, the model has better performance when the kernel is polynomial or rbf. So next we will use the SVM model with kernel as rbf.

# In[62]:


model_svm = svm.SVC(C = 0.8, kernel='rbf')
fit_svm = model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(y_test, y_predict)))

print(pd.DataFrame(confusion_matrix(y_test,y_predict)))
plot_confusion_matrix(confusion_matrix(y_test,y_predict), classes=class_names, normalize = True, title='Normalized Confusion Matrix: SVM')


# ### Cross validation of SVM

# In[63]:


scores = cross_val_score(model_svm, X_test, y_test, cv=5, scoring='f1')
print("SVM: " + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[66]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_lr, X_test, y_test, cv=25, scoring='f1')

print("Logistic regression: " + "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[ ]:




