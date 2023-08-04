import pandas as pd

def get_timestamp_at_index(csv_file_path, index_to_find):
     # Read only the 'DateT' column
    df = pd.read_csv(csv_file_path, usecols=['DateT'], error_bad_lines=False)

    # Retrieve the DateT value at the specified index
    timestamp = df.loc[index_to_find, 'DateT']
    return timestamp

# Example usage:
# csv_file = "path/to/yourfile.csv"
# index_to_find = 332
# timestamp = get_timestamp_at_index(csv_file, index_to_find)
# print(f"The timestamp at index {index_to_find} is {timestamp}")
