import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def random_forest(dataset):
    features = dataset.iloc[:, 1:len(dataset.columns)-2].values
    prediction = dataset.iloc[:, len(dataset.columns)-1].values
    features_train, features_test, prediction_train, prediction_test = train_test_split(features, prediction, test_size=0.2, random_state=0)

    # Feature Scaling (npt sure if I need this because all the features are in the same scale I think)
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    features_train = sc.fit_transform(features_train)
    features_test = sc.transform(features_test)

    from sklearn.ensemble import RandomForestClassifier

    regressor = RandomForestClassifier(n_estimators=5, random_state=0)
    regressor.fit(features_train, prediction_train)
    important_features_dict = {}
    for idx, val in enumerate(regressor.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    print(f'5 most important features: {important_features_list[:5]}')
    x = regressor.feature_importances_
    y_pred = regressor.predict(features_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(f'{confusion_matrix(prediction_test, y_pred)}')
    print(f'{classification_report(prediction_test, y_pred)}')
    print(f'{accuracy_score(prediction_test, y_pred)}')


def main():
    from datetime import datetime
    start = datetime.now()

    cwd = os.getcwd()
    dataset = pd.read_csv(f'{cwd}/data_strong.csv')
    print(f'rows: {len(dataset.index)},columns: {len(dataset.columns)}')
    print(f'time:{datetime.now() - start}')
    random_forest(dataset)


if __name__ == '__main__':
    main()
