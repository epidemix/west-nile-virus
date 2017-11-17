import os
import process_data as process
import predict_wnv as predict
import train_model as train
import spray_recommendations as spray


def main():
    print('Running model...')
    process.main()
    if not os.path.isfile('models/wnv_predict.pkl'):
        train.main()
    predict.main()
    spray.main()
    print('Done.')


if __name__ == "__main__":
    main()
