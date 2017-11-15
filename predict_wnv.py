import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
import predict_mosquitos


def main():
    model = joblib.load('wnv_predict.pkl')
    # try:
    #     model_2 = joblib.load('mosquito_predict.pkl')
    # except:
    #     print('building predictions of mosquitos...\n')
    #     predict_mosquitos.main()
    #     model_2 = joblib.load('mosquito_predict.pkl')

    data = pd.read_csv("/repos/epidemix/west-nile-virus/processed_data/processed_test.csv")
    data = data.drop(['date', 'unspecified_culex'], axis=1)
    data['id'] = data['id'] + 1
    data.set_index('id', inplace=True)

    # mosquitoes = model_2.predict(data)
    # data['nummosquitos'] = mosquitoes

    print("predict wnv: \n", data.columns)

    predictions = model.predict(data)
    print(predictions.shape)

    submission = pd.DataFrame(predictions, columns=['WnvPresent'])
    submission.index.rename('Id', inplace=True)
    submission.index = submission.index + 1
    submission.to_csv('submission_data/submission.csv')
    print("Done.")


if __name__ == "__main__":
    main()
