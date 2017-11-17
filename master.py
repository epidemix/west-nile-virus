import os
import process_data as process
import predict_wnv as predict
import train_model as train
import spray_recommendations as recommend


def main():
    """
        Run the west nile virus prediction, and intervention recommendation model.
    """
    print('Running model...')
    process.main()                                    # process data
    if not os.path.isfile('models/wnv_predict.pkl'):  # check if trained model exists
        train.main()                                  # train model if one does not exist
    predict.main()                                    # generate wnv predictions
    recommend.main()                                  # generate recommendations
    print('Done.')


if __name__ == "__main__":
    main()
