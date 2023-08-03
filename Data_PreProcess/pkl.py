import pickle as pkl
import pandas as pd

def unpickle(pickle_file, unpickled_file_name):
    with open(pickle_file, "rb") as f:
        object = pkl.load(f)
        
    df = pd.DataFrame(object)
    df.to_csv(unpickled_file_name)
    
# Example usage:
pickle_file = "adj_mx.pkl"
unpickled_file_name = "unpickled_data.csv"
unpickle(pickle_file, unpickled_file_name)
