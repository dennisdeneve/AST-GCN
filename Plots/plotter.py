import matplotlib.pyplot as plt
import yaml
import os

def create(model, config):
    print("Creating box and whiskers plot")
    metrics = {'MSE': {3:[],6:[],9:[],12:[],24:[]},'MAE': {3:[],6:[],9:[],12:[],24:[]},'RMSE': {3:[],6:[],9:[],12:[],24:[]},'SMAPE': {3:[],6:[],9:[],12:[],24:[]}}
    horizons = config['forecasting_horizons']['default']
    stations = config['stations']['default']
    # Iterate over each station
    for station in stations:
        # Iterate over each forecasting horizon
        for horizon in horizons:
            try:
                metric_file = f'Results/{model}/{horizon} Hour Forecast/{station}/Metrics/metrics.txt'
                with open(metric_file, 'r') as file:
                    lines = file.readlines()   
                for line in lines:
                    # New Parsing Logic
                    if "MSE:" in line:
                        value = float(line.split("MSE:")[1].strip())
                        metrics["MSE"][horizon].append(value)
                    elif "RMSE:" in line:
                        value = float(line.split("RMSE:")[1].strip())
                        metrics["RMSE"][horizon].append(value)
                        print("RMSE values:", metrics["RMSE"])
                    elif "MAE:" in line:
                        value = float(line.split("MAE:")[1].strip())
                        metrics["MAE"][horizon].append(value)
                    elif "SMAPE:" in line:
                        value = float(line.split("SMAPE:")[1].strip())
                        metrics["SMAPE"][horizon].append(value)
            except Exception as e:
                print(e)
                print('Error! : Unable to read data or write metrics for station {} and forecast length {}. Please review the data or code for the metrics for errors.'.format(station, horizon))
    keys = metrics.keys()
    box_colors = ['blue', 'orange', 'green', 'red', 'purple']
    for key in keys:
        # Create a figure and axis
        fig, ax = plt.subplots()
        # Create the box and whisker plot
        boxplot = ax.boxplot([metrics[key][h] for h in [3, 6, 9, 12, 24]], patch_artist=True)
        # Set the colors for each box individually
        for box, color in zip(boxplot['boxes'], box_colors):
            box.set(facecolor=color)
        # Set the title and labels
        ax.set_title(model)
        ax.set_xlabel('HORIZONS')
        ax.set_ylabel(key)
        # Set the x-axis ticks and labels
        x_values = [3, 6, 9, 12,24]
        ax.set_xticks(range(1, len(x_values) + 1))
        ax.set_xticklabels(x_values)

        # Set the color of the median line
        median_color = 'black'  # Change to the desired color
        medianprops = dict(color=median_color)
        plt.setp(ax.lines, **medianprops)
        # Show the plot
        # plt.show()
        # Ensure directory exists and save the plot
        os.makedirs(f'Plots/{model}', exist_ok=True)
        plt.savefig(f'Plots/{model}/{key}_BWplot.png')