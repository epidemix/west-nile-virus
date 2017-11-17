import pandas as pd
from sklearn.externals import joblib


def main():
    """
        Generate West Nile Virus Predictions
    """
    print("Predicting skeeters...")
    model = joblib.load('models/wnv_predict.pkl')            # load model

    # Read in the 'Test' data
    data = pd.read_csv("processed_data/processed_test.csv")
    data = data.drop(['date', 'unspecified_culex'], axis=1)
    data['id'] = data['id'] + 1
    data.set_index('id', inplace=True)

    # Predicting West Nile Virus
    predictions = model.predict(data)                        # binary predictions
    probabilities = model.predict_proba(data)                # probability predictions

    # write kaggle submission data
    submission = pd.DataFrame(predictions,
                              columns=['WnvPresent'])
    submission.index.rename('Id', inplace=True)
    submission.index = submission.index + 1
    submission.to_csv('results/kaggle_submission.csv')

    # store positive probabilities
    probs = pd.DataFrame(probabilities[:,1],
                         columns=['wnv_probablitiy'])
    probs.index.rename('Id', inplace=True)
    probs.index = probs.index + 1
    probs.to_csv('results/prediction_probabilities.csv')

    print("Predictions complete.")


if __name__ == "__main__":
    main()
