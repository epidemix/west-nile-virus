import os
import predict_wnv as predict
import train_model as train


def main():
    if not os.path.isfile('wnv_predict.pkl'):
        train.main()
    predict.main()


if __name__ == "__main__":
    main()
