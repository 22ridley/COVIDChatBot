from pandas import *
import warnings
import pickle
from numpy import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from dataprep.eda import *
from dataprep.eda.missing import plot_missing
from dataprep.eda import plot_correlation

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    pandas.set_option('display.expand_frame_repr', False)
    url = "symptoms_COVID.csv"
    header = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore Throat', 'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue', 'Gastrointestinal', 'Travel Abroad', 'Contact with COVID Patient', 'Attended Large Gathering', 'Visited Public Exposed Places', 'Family Working in Public Exposed Places', 'Wearing Masks', 'Sanitization from Market', 'COVID-19']

    df = read_csv(url, names=header, low_memory=False)
    df = df.iloc[1:, :]

    e = LabelEncoder()

    df['Breathing Problem'] = e.fit_transform(df['Breathing Problem'])
    df['Fever'] = e.fit_transform(df['Fever'])
    df['Dry Cough'] = e.fit_transform(df['Dry Cough'])
    df['Sore Throat'] = e.fit_transform(df['Sore Throat'])
    df['Running Nose'] = e.fit_transform(df['Running Nose'])
    df['Asthma'] = e.fit_transform(df['Asthma'])
    df['Chronic Lung Disease'] = e.fit_transform(df['Chronic Lung Disease'])
    df['Headache'] = e.fit_transform(df['Headache'])
    df['Heart Disease'] = e.fit_transform(df['Heart Disease'])
    df['Diabetes'] = e.fit_transform(df['Diabetes'])
    df['Fatigue'] = e.fit_transform(df['Fatigue'])
    df['Travel Abroad'] = e.fit_transform(df['Travel Abroad'])
    df['Contact with COVID Patient'] = e.fit_transform(df['Contact with COVID Patient'])
    df['Attended Large Gathering'] = e.fit_transform(df['Attended Large Gathering'])
    df['Visited Public Exposed Places'] = e.fit_transform(df['Visited Public Exposed Places'])
    df['Wearing Masks'] = e.fit_transform(df['Wearing Masks'])

    df['COVID-19'] = e.fit_transform(df['COVID-19'])
    df['Sanitization from Market'] = e.fit_transform(df['Sanitization from Market'])
    df['Gastrointestinal'] = e.fit_transform(df['Gastrointestinal'])
    df['Hyper Tension'] = e.fit_transform(df['Hyper Tension'])
    df['Family Working in Public Exposed Places'] = e.fit_transform(df['Family Working in Public Exposed Places'])


    df = df.drop(columns=['Sanitization from Market', 'Gastrointestinal', 'Hyper Tension', 'Family Working in Public Exposed Places'], axis=1)
    x = df.iloc[:, :-1]
    y = df['COVID-19']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    '''t = tree.DecisionTreeClassifier()
    t.fit(x_train, y_train)
    y_pred = t.predict(x_test)'''

    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, x_train, y_train)
    t = clf.fit(x_train, y_train)

    filename = 'COVID_model.sav'
    pickle.dump(t, open(filename, 'wb'))

    accuracy = t.score(x_test, y_test) * 100
    print(accuracy)