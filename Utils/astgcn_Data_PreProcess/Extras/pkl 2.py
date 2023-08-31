import pickle as pkl
import pandas as pd

def unpickle(pickle_file, unpickled_file_name):
    with open(pickle_file, "rb") as f:
        object = pkl.load(f)
    # Reshape 'a' to be a 2D array, -1 makes numpy calculate the correct number for that dimension
    a_2d = object.reshape(a.shape[0], -1)

    # Now you can create a DataFrame
    df = pd.DataFrame(a_2d)	    
    # df = pd.DataFrame(object)
    df.to_csv(unpickled_file_name)
    
# Example usage:
pickle_file = "outputs_0.pkl"
unpickled_file_name = "unpickled_data_GWN.csv"
unpickle(pickle_file, unpickled_file_name)
