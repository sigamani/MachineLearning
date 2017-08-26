from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

# Import some classifiers to test
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import linear_model, datasets
from sklearn.naive_bayes import GaussianNB

# Calculate the P-R curve for each classifier
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.cross_validation import train_test_split
import pydot


features      = ['BkToPr','DivYld','EngsYld','SalesGr',
                 'AssToEq','MCap','Beta','DbtToEq','1YrVol','5YrVol',
                 '3YrVol','ExpToCurr','MomST','MomLT','Sent',
                 'RtnOnEq','TotATrn','SecID']

py.sign_in('michael.sigamani_sr', 'pUwzlMuStskB3R2zemzc')


# np.set_printoptions(threshold=np.inf)

# =====================================================================

def download_data(URL):
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    frame = read_table(
        URL,
        

        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
#        skipinitialspace=True,

        # Generate row labels from each row number
#        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        #header=None,
        header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame

    print("Downloading data from {}".format(URL))

    return frame

# =====================================================================


def get_features_and_labels(frame):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0, or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
       
    # Use 80% of the data for training; test against the rest
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test


def get_features_and_labels2(frame):
    
    arr = np.array(frame, dtype=np.float)
    X, y = arr[:, :-1], arr[:, -1]

    return X, y


def get_features_and_labels3(frame):
    
    arr = np.array(frame, dtype=np.float)
    X, y = arr[:, :-1], arr[:, -1]

    return X, y


# =====================================================================


def evaluate_classifier(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    '''
    
    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.4, gamma=1e-3)
    # Fit the classifier
    classifier.fit(X_train, y_train)

    score = f1_score(y_test, classifier.predict(X_test))



    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    base_classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, criterion="entropy")
    base_classifier.fit(X_train,y_train)
    tree.export_graphviz(base_classifier, out_file=r'C:\\Users\\michael\\tree.dot', feature_names = features) 


    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(base_classifier, n_estimators=500, learning_rate=1.0, algorithm='SAMME.R')    
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))

    #getRanking(classifier, X_train)

    # Generate the P-R curve
    # classifier.
    y_prob = classifier.decision_function(X_test)
    y_prob1 = classifier.predict_proba(X_test)
    #y_prob1 = classifier.decision_function(X_train)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Boosted Decision Tree (F1 score={:.3f})'.format(score), precision, recall
    #print(y_prob)
    makeResponsePlot(y_prob)

	# Test a logistic regression classifier 
    classifier = linear_model.LogisticRegression(C=1e5)
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
   
   
    # Generate the P-R curve
    # classifier.
    y_prob = classifier.decision_function(X_test)
   # makePlot(y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Logistic Regression (F1 score={:.3f})'.format(score), precision, recall

    # Test a naive baes classifier 
    #classifier = GaussianNB()
    #classifier.fit(X_train,y_train)
    #score = f1_score(y_test, classifier.predict(X_test))

    # Generate the P-R curve
    #y_prob = classifier.decision_function(X_test)
    #precision, recall, _ = precision_recall_curve(y_test, y_prob)
    #yield 'Naive Bayes (F1 score={:.3f})'.format(score), precision, recall

     
# =====================================================================

def getRanking(classifier, X):

    importances = classifier.feature_importances_ 
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
   
    sortedfeaturelist = [i[0] for i in sorted(zip(features, indices), key=lambda l: l[1], reverse=True)]
 
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure(num=None, figsize=(5, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.xticks(rotation=90)
    plt.title("Feature importances (Sep-Oct 2016)")
    plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), sortedfeaturelist)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    #data = [go.Histogram(x=importances)]
    #py.iplot(data, filename='FeatureImportance')
    

# =====================================================================

def plot(results):

    fig = []
    for label, precision, recall in results:

        data0 = go.Scatter(x = recall, y = precision, name = label)
        fig.append(data0)

    layout = go.Layout(
    title="Precision-Recall Curve",
    yaxis = dict( range=[0.5, 0.7],
                 title='Recall',
                 titlefont=dict(
                 family='Courier New, monospace',
                 size=18,
                 color='#7f7f7f')),
    xaxis = dict(title='Precision',
                 titlefont=dict(
                 family='Courier New, monospace',
                 size=18,
                 color='#7f7f7f'))
    )
    

    fig = go.Figure(data = fig, layout=layout)
    py.iplot(fig, filename="test")




def makeResponsePlot(input):

  #  x = np.random.randn(500)
    data = [go.Histogram(x=input, xbins=dict(
        start=-2.0,
        end=2.0,
        size=0.1
    ))]
    layout = go.Layout(

        title='Feb-Mar 2016 (negative return)',
        xaxis=dict(
            title='BDT response',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Entries',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='Data-B')
    


if __name__ == "__main__":

    URL1 = "http://sigamani.com/MachineLearningData/JanFeb2016.csv"
    URL2 = "http://sigamani.com/MachineLearningData/FebMar2016.csv"

    frame = download_data(URL1)
    frame2 = download_data(URL2)

    # Process data into feature and label arrays
    print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    
    #In-sample testing     
    #X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Out-of-sample testing 
    X_train, y_train = get_features_and_labels2(frame)
    X_test, y_test = get_features_and_labels2(frame2)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    plot(results)
 
    #(graph,) = pydot.graph_from_dot_file(r'C:\\Users\\michael\\tree.dot')
    #graph.write_png(r'C:\\Users\\michael\\tree.png')