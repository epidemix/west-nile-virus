import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
import numpy as np


def main():
    df = pd.read_csv('processed_data/processed_train.csv').set_index('date')
    y = df.pop('nummosquitos')
    x = df.drop('wnvpresent', axis=1)
    x = normalize(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    forest = RandomForestRegressor(n_jobs=-1)

    print(x_train.columns)

    param_grid = dict(max_depth=np.random.randint(1, 10, 5),
                      min_samples_split=np.random.sample(5),
                      min_samples_leaf=np.random.sample(5) / 2,
                      min_weight_fraction_leaf=np.random.sample(5) / 2,
                      max_features=np.random.sample(5))

    grid = GridSearchCV(forest,
                        param_grid,
                        cv=10,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=True)

    grid.fit(x_train, y_train)
    forest = grid.best_estimator_

    print("Mosquito Estimator:\n", forest)
    joblib.dump(forest, 'mosquito_predict.pkl')


if __name__ == "__main__":
    main()
