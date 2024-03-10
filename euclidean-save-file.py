import pickle
import pandas as pd

load_path = "/content/drive/My Drive/euclidean_distance.pkl"
save_csv_path = "/content/drive/My Drive/euclid_distances.csv"

try:
    with open(load_path, "rb") as file:
        data = pickle.load(file)

        # Saving data into DataFrame
        df = pd.DataFrame(data[3], columns=[str(i) for i in range(len(data[3]))])

        # Saving DataFrame to CSV file
        df.to_csv(save_csv_path, index=False)

    print(f"Data successfully saved to {save_csv_path}")
except (FileNotFoundError, EOFError) as e:
    print(f"Error: {e}")
