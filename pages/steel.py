# Import standard libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import scikit-learn libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree  


# Creating the top contents
webpage_title = 'Iris Classification using Decision Trees'

webpage_definition = (""" 
                      
    The Iris dataset was used in R.A. Fisher's classic 1936 paper, 
    the Use of Multiple Measurements in Taxonomic Problems, 
    and can also be found on the UCI Machine Learning Repository.
    
    It includes three iris species with 50 samples each as well as some 
    properties about each flower. One flower species is linearly separable 
    from the other two, but the other two are not linearly separable from 
    each other.
   
     """)
       
def top_contents(title, definition):
    
    st.title(title)
    st.write(definition)
    st.markdown("---")
    
    return

introduction = top_contents(webpage_title, webpage_definition)



filename = 'Iris.csv'

def raw_dataset(file_name):
    st.header('View of the dataset')   
    df = pd.read_csv('Iris.csv')
    st.dataframe(df
)    
    return df
    
dict_iris = {
    
    'Id': 'Numeric sequence',
    'SepalLengthCm': 'Length of the sepal (in cm)',
    'SepalWidthCm': 'Width of the sepal (in cm)',
    'PetalLengthCm': 'Length of the petal (in cm)',
    'PetalWidthCm': 'Width of the petal (in cm)',
    'Species': 'Species name'

    }

feats = st.selectbox('Select a feature:', list(dict_iris.keys()))

rr1, rr2 = st.columns([1,8])

with rr1:
    st.markdown('**Meaning:**')

with rr2:
    st.write(dict_iris.get(feats))


dataset = raw_dataset(filename)
    
# View the null values for each column
null_values, quick_summaries = st.columns((3, 5))

with null_values:
    st.markdown('#### Columnar Null Values')
    nulls = dataset.isna().sum()
    st.dataframe(nulls)

with quick_summaries:
    st.markdown('### Quick Summaries')
    rows, columns = dataset.shape
    classes = dataset['Species'].unique()
    st.write("Number of columns:", columns)
    st.write("Number of Rows", rows)
    st.write("Number of classes:", len(classes))
    st.write(classes)

#%% # Viewing the dataframe
# st.markdown("#### View of the dataset")
# df = pd.read_csv('Iris.csv')
# st.dataframe(df)



    


st.markdown('# Now for the scikit part....')


target, features = st.columns((0.5, 3))


le = LabelEncoder()
dataset['labels'] = le.fit_transform(dataset["Species"])
X = np.array(dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])                   # Input
Y = np.array(dataset[["labels"]])   


with target:
    st.dataframe(dataset['labels'])

with features:
    st.dataframe(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 100)                                    # 80:20

st.markdown('## Decision Tree')                                                                        

classifier_type = st.selectbox('Select Criterion', ['gini', 'entropy'])

def dtc_parameter(selectbox):
    params = dict()
    if selectbox == 'Gini':
        splitter = st.select_slider('Splitter', ['Best', 'Random'])
        max_depth = st.slider('Depth of tree', 0, 100, 1)
        min_samples_split = st.slider('Min # of samples', 0, 20, 2)
        min_samples_leaf = st.slider('Min # of samples', 0, 20, 1)
        
        params['Split Type'] = splitter
        params['Max Depth'] = max_depth
        params['Split Samples'] = min_samples_split
        params['Leaf Samples'] = min_samples_leaf
    else:
        splitter = st.select_slider('Splitter for Entropy', ['best', 'random'])
        max_depth = st.slider('Depth of tree', 0, 100, 1)
        min_samples_split = st.slider('Min # of samples', 0, 20, 2)
        min_samples_leaf = st.slider('Min # of samples', 0, 20, 1)
        
        params['Split Type'] = splitter
        params['Max Depth'] = max_depth
        params['Split Samples'] = min_samples_split
        params['Leaf Samples'] = min_samples_leaf
    
    return params

paramsdtc = dtc_parameter(classifier_type)


result = st.selectbox('Whatchu want boi', ['Train Model', 'Visualize Model'])

# clf_gini = DecisionTreeClassifier(criterion = classifier_type,
#                                   splitter = params['Split Type']
#                                   params['Max Depth'] = 
#                                   params['Split Samples'] = min_samples_split
#                                   params['Leaf Samples'] = min_samples_leaf

def dtc_actions(selectbox, params, classifier_type, X_train, y_train, X_test):
    if selectbox == 'Train Model':
        clf = DecisionTreeClassifier(criterion = classifier_type,
                                     splitter = params['Split Type'],
                                     max_depth = params['Max Depth'],
                                     min_samples_split = params['Split Samples'],
                                     min_samples_leaf = params['Leaf Samples'],
                                     )
      
        clf.fit(X_train, y_train)
        
        dtc_pred = clf.predict(X_test)
        
        return st.dataframe(dtc_pred)
        
    else:
        clf = DecisionTreeClassifier(
            criterion = classifier_type,
            splitter = params['Split Type'],
            max_depth = params['Max Depth'],
            min_samples_split = params['Split Samples'],
            min_samples_leaf = params['Leaf Samples'])
      
        clf.fit(X_train, y_train)
        
        fig = plt.figure(figsize=(25,20))
        fff = tree.plot_tree(clf, 
                   filled=True)
        
        plot = tree.plot_tree(clf)
        
        return st.pyplot(fig)

st.sidebar.title('HAys Iris')

dtc_final = dtc_actions(result, paramsdtc, classifier_type, 
                        X_train, Y_train, X_test)



