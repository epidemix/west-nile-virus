# Predicting West Nile Virus in Chicago
## GA DSI-6: Project 4

[Here](https://www.kaggle.com/c/predict-west-nile-virus) is a link to the related kaggle page.

Program Files:
- `master.py`
- `process_data.py`
- `train_model.py`
- `predict_wnv.py`
- `spray_recommendations.py`

Data:
- `data/train.csv`
- `data/test.csv`
- `data/geography.csv`
- `data/demographics.csv`
- `data/traps.csv`

Generated Data:
- `results/kaggle_submission.csv`
- `results/prediction_probabilities.csv`
- `spray_recommendations/aggressive_recommendation.csv`
- `spray_recommendations/moderate_recommendation.csv`
- `spray_recommendations/conservative_recommendation.csv`

Use:

To use the prediction system, first run the file `process_data.py` to prepare the data sets.
Then run the file `master.py`.
Once this program has completed, the mosquito predictions can be found in the `results` directory.
The advised spray recommendations can then be found in the `spray_recommendations` directory. 
