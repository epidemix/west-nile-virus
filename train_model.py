import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import process_data as process


def main():
    data_file = "data/processed_train.csv"
    try:
        model_data = pd.read_csv(data_file)
    except:
        print('Processed data missing.\nProcessing raw data.')
        process.main()
        model_data = pd.read_csv(data_file)

    model_data = model_data.drop(['date', 'nummosquitos'], axis=1)
    y = model_data.pop('wnvpresent')
    X = model_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    print(X_train.columns)

    forest = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=42)

    param_grid = dict(max_depth=np.random.randint(1, 10, 5),
                      min_samples_split=np.random.sample(5),
                      min_samples_leaf=np.random.sample(5)/2,
                      min_weight_fraction_leaf=np.random.sample(5)/2,
                      max_features=np.random.sample(5))

    grid = GridSearchCV(forest,
                        param_grid,
                        cv=10,
                        scoring='neg_log_loss',
                        n_jobs=-1,
                        verbose=True)

    grid.fit(X_train, y_train)
    forest = grid.best_estimator_
    print(forest)
    joblib.dump(forest, 'wnv_predict.pkl')


if __name__ == "__main__":
    main()
