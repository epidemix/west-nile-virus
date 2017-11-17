import pandas as pd
from sklearn.externals import joblib


def main():
    model = joblib.load('wnv_predict.pkl')

    data = pd.read_csv("/repos/epidemix/west-nile-virus/processed_data/processed_test.csv")
    data = data.drop(['date', 'unspecified_culex'], axis=1)
    data['id'] = data['id'] + 1
    data.set_index('id', inplace=True)

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    submission = pd.DataFrame(predictions, columns=['WnvPresent'])
    submission.index.rename('Id', inplace=True)
    submission.index = submission.index + 1
    submission.to_csv('submission_data/kaggle_submission.csv')

    probs = pd.DataFrame(probabilities[:,1], columns=['wnv_probablitiy'])
    probs.index.rename('Id', inplace=True)
    probs.index = probs.index + 1
    probs.to_csv('data/prediction_probabilities.csv')

    print("Done.")


if __name__ == "__main__":
    main()
