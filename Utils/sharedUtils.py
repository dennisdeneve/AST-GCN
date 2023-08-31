import numpy as np
import os

def create_file_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    # print(f"Creating directory: {directory}")  # print for debugging
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()

        
def sliding_window(df, lag, forecast, split, set, n_stations):
    """
    Converts array to times-series input-output sliding-window pairs.
    Parameters:
        df - DataFrame of weather station data
        lag - length of input sequence
        forecast - length of output sequence(forecasting horizon)
        split - points at which to split data into train, validation, and test sets
        set - indicates if df is train, validation, or test set
    Returns:
        x, y - returns x input and y output
    """
    if set == 0:
        samples = int(split[0] / n_stations)
    if set == 1:
        samples = int(split[1] / 45 - split[0] / n_stations)
    if set == 2:
        samples = int(split[2] / 45 - split[1] / n_stations)


    dfy = df.drop(['Rain', 'Humidity', 'Pressure', 'WindSpeed', 'WindDir'], axis=1)
    stations = n_stations
    features = 6

    df = df.values.reshape(samples, stations, features)
    dfy = dfy.values.reshape(samples, stations, 1)

    x_offsets = np.sort(np.concatenate((np.arange(-(lag - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(1, (forecast + 1), 1))

    data = np.expand_dims(df, axis=-1)
    data = data.reshape(samples, 1, stations, features)
    data = np.concatenate([data], axis=-1)

    datay = np.expand_dims(dfy, axis=-1)
    datay = datay.reshape(samples, 1, stations, 1)
    datay = np.concatenate([datay], axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(samples - abs(max(y_offsets)))  # Exclusive

    # t is the index of the last observation.
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(datay[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x = np.squeeze(x)
    y = np.squeeze(y, axis=2)
    return x, y