import pandas as pd
import numpy as np


def get_lags(data, x, lag, f=None):
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


def main():
    for file in ['train.csv', 'test.csv']:
        data_file = file
        train = pd.read_csv('data/' + data_file)
        species_dummies = pd.get_dummies(train['Species'])
        df = train.join(species_dummies)
        df.rename(columns={c: c.lower().replace(' ', '_') for c in df.columns}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].apply(lambda x: x.isocalendar()[1])
        
        spray_traps = pd.read_csv('data/traps.csv')
        spray_traps['Date'] = pd.to_datetime(spray_traps['Date'])
        spray_traps['year'] = spray_traps['Date'].dt.year
        spray_traps['week'] = spray_traps['Date'].apply(lambda x: x.isocalendar()[1])
        summed = spray_traps.drop('spray_distance', axis=1).groupby(['trap', 'year', 'week']).sum().reset_index()
        meaned = spray_traps.drop('spray', axis=1).groupby(['trap', 'year', 'week']).mean().reset_index()
        spray_traps = summed.merge(meaned)
        df = df.merge(spray_traps, on=['year', 'week', 'trap'], how='left')
        df['spray'] = df['spray'].replace(np.NaN, 0.0)
        df['spray_distance'] = df['spray_distance'].replace(np.NaN, 0.0)
        
        weather = pd.read_csv('data/weather.csv')
        weather.Date = pd.to_datetime(weather.Date)
        weather['year'] = weather['Date'].dt.year
        weather['week'] = weather['Date'].apply(lambda x: x.isocalendar()[1])
        df_2 = pd.DataFrame()

        df_2['ten_day_avg_percip'] = get_lags(weather, 'PrecipTotal', 30, f='sum')
        df_2['ten_day_avg_temp'] = get_lags(weather, 'Tavg', 10)
        df_2['ten_day_avg_dewpoint'] = get_lags(weather, 'DewPoint', 10)
        df_2['ten_day_avg_pressure'] = get_lags(weather, 'StnPressure', 10)
        df_2['ten_day_avg_windspeed'] = get_lags(weather, 'AvgSpeed', 10)
        weather_dates = pd.DataFrame(weather.Date)
        w = weather_dates.join(df_2)
        w = w.groupby('Date').mean().reset_index()
        w = w.rename(columns={c: c.lower() for c in w.columns})
        model_data = df.merge(w, on="date", how='left').set_index('date')

        model_data = model_data.drop(['address', 'species', 'block', 'street',
                                      'trap', 'addressnumberandstreet', 'addressaccuracy'], axis=1)


        model_data.to_csv('processed_data/processed_' + data_file)

    print('Data processing complete.')


if __name__ == "__main__":
    main()
