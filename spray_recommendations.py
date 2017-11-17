import pandas as pd
import numpy as np


def get_lags(data, x, lag, f=None):
    """
        Generate lagged variables

    :param data: object that contains the data
    :type data: pandas.DataFrame
    :param x: variable of interest
    :type x: str
    :param lag: number of periods for the lag
    :type lag: int
    :param f: function for aggregation
    :type f: func
    :return: lagged variables
    :rtype: pandas.Series
    """
    lag_avg = pd.DataFrame(data[x])

    def assign_na(x):
        try:
            y = np.float64(x)
        except:
            y = np.NaN
        return y

    lag_avg[x] = lag_avg[x].apply(assign_na)

    for i in range(1, lag):
        lag_avg['lag_{}'.format(i)] = lag_avg[x].copy().shift(i)

    ten_day_avg = []
    for i, row in lag_avg.iterrows():
        if f == 'sum':
            ten_day_avg.append(row.sum(skipna=True))
        else:
            ten_day_avg.append(row.mean(skipna=True))

    return pd.Series(ten_day_avg)


def main(cost=2500):
    print('Generating policy recommendations...')
    spray_cost = cost  # per square mile

    df_in = pd.read_csv('data/train.csv')

    zip_codes = pd.read_csv('data/zip_codes.csv')

    zip_codes = zip_codes[['Trap', 'ZipCode']].rename(columns={'Trap': 'trap',
                                                               'ZipCode': 'zip_code'})

    df_out = df_in.join(zip_codes)

    demographics = pd.read_csv('data/demographics.csv').drop('Unnamed: 0', axis=1)

    dfsss = df_out.merge(demographics, on=['zip_code'])

    dfsss.drop('trap', axis=1, inplace=True)

    to_drop = ['Address', 'Species', 'Block', 'Street', 'AddressNumberAndStreet',
               'AddressAccuracy', 'WnvPresent', 'Trap']

    dfsss = dfsss.drop(to_drop, axis=1)

    geo = pd.read_csv('data/geography.csv')
    geo = geo[['ZipCode', 'LandArea', 'WaterArea']]
    geo = geo.groupby('ZipCode').mean().reset_index().rename(columns={'ZipCode': 'zip_code'})

    dfsss = dfsss.merge(geo, on='zip_code')
    dfsss.rename(columns={'LandArea': 'land_area',
                          'WaterArea': 'water_area'}, inplace=True)

    to_sum = dfsss[['Date', 'zip_code', 'NumMosquitos']]
    summed = to_sum.groupby(['Date', 'zip_code']).sum().reset_index()

    to_mean = dfsss[['Date', 'zip_code', 'Latitude', 'Longitude']]
    meaned = to_mean.groupby(['Date', 'zip_code']).mean().reset_index()

    df = meaned.merge(summed, on=['Date', 'zip_code'])
    df = df.merge(dfsss, on=['Date', 'zip_code'], how='left')
    df = df.drop(['Latitude_y', 'Longitude_y', 'NumMosquitos_y'], axis=1)
    df = df.rename(columns={'Date': 'date',
                            'Latitude_x': 'latitude',
                            'Longitude_x': 'longitude',
                            'NumMosquitos_x': 'num_mosquitos'})

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].apply(lambda x: x.isocalendar()[1])

    wnv_years = []
    for i in range(2002, 2017):
        wnv_years.append(str(i))
    wnv_cases = [635, 20, 23, 135, 29, 23, 9, 1, 30, 22, 174, 60, np.NaN, 27, 90]
    wnv_deaths = [42, 1, 2, 6, np.NaN, 0, 0, 0, 1, 1, 5, 7, np.NaN, 0, 0]

    deaths = pd.DataFrame({"Year": wnv_years, "Cases": wnv_cases, "Deaths": wnv_deaths})
    deaths = deaths.set_index('Year')
    deaths = deaths.shift(1).reset_index()
    deaths = deaths.rename(columns={'Year': 'year', 'Cases': 'averaged_cases', 'Deaths': 'averaged_deaths'})
    deaths['year'] = deaths['year'].astype(np.int64)

    deaths['averaged_cases'] = get_lags(deaths, 'averaged_cases', 5, np.mean)
    deaths['averaged_deaths'] = get_lags(deaths, 'averaged_deaths', 5, np.mean)
    df = df.merge(deaths, on='year')

    # create metric variables
    df['at_risk_population_percent'] = df['percent_age_over_65'] + df['percent_age_under_14']
    df['at_risk_population'] = df['at_risk_population_percent'] / 100 * df['total_population']
    df['proportional_averaged_cases'] = df['population_porportion'] * df['averaged_cases']
    df['proportional_averaged_deaths'] = df['population_porportion'] * df['averaged_deaths']
    df['high_risk_infection_percent'] = df['averaged_cases'] * df['at_risk_population_percent'] / 100

    df = df.groupby(['date', 'zip_code']).mean().reset_index()

    to_drop_2 = ['percent_employed', 'percent_no_health', 'percent_public_health',
                 'percent_age_over_65', 'percent_age_under_14', 'percent_low_income']

    df = df.drop(to_drop_2, axis=1)

    predictions = pd.read_csv('results/prediction_probabilities.csv')
    predictions['Id'] -= 1
    predictions = predictions.set_index('Id')

    # recombine the predictions
    df = df.join(predictions)

    df['cost_to_spray'] = spray_cost * df['land_area']
    df['spray_metric'] = df['wnv_probablitiy'] * df['at_risk_population'] / df['cost_to_spray'] * 100

    # isolate essential variables
    df = df[['date', 'zip_code', 'wnv_probablitiy', 'cost_to_spray', 'spray_metric']]

    aggressive = df[df['spray_metric'] > 6]
    moderate = df[df['spray_metric'] > 7]
    conservative = df[df['spray_metric'] > 8]

    # write recommendation data
    aggressive.to_csv('spray_recommendations/aggressive_recommendation.csv')
    moderate.to_csv('spray_recommendations/moderate_recommendation.csv')
    conservative.to_csv('spray_recommendations/conservative_recommendation.csv')
    print('Recommendation files written.')


if __name__ == "__main__":
    main()
