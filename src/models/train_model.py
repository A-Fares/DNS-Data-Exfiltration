import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.visualization.visualize import plot_confusion_matrix, get_model_evaluation
from sklearn.model_selection import train_test_split
import pickle
from src.features.build_features import preprocessing_data



def training_model(df):
    '''
     get the categorical column and drop all of them,
      that not affect on the training data
    '''
    df = preprocessing_data(df)

    # splitting features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ## train the classifier
    rf_cls = RandomForestClassifier(n_estimators=100)
    rf_cls.fit(X_train.values, y_train.values)
    y_pred = rf_cls.predict(X_test)

    # evaluate the model
    get_model_evaluation(y_test, y_pred)

    ## Visualize the confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    ## save the model
    with open(r"..\models\model_rf.sav", 'wb') as file:
        pickle.dump(rf_cls, file)
