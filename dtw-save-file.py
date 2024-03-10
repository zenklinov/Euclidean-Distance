import pickle
import pandas as pd

load_path = "/content/drive/My Drive/dtw_distances.pkl"
save_csv_path = "/content/drive/My Drive/dtw_distances.csv"

try:
    with open(load_path, "rb") as file:
        data = pickle.load(file)

        # Generate meaningful column names
        column_names = [f'Distance_{i+1}' for i in range(len(data[3]))]

        # Save data into DataFrame
        df = pd.DataFrame(data[3], columns=column_names)

        # Save DataFrame to CSV file
        df.to_csv(save_csv_path, index=False)

    print(f"Data successfully saved to {save_csv_path}")
except (FileNotFoundError, EOFError) as e:
    print(f"Error: {e}")
