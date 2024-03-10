from google.colab import drive
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# File path relative to the mounted Google Drive
file_path = '/content/drive/My Drive/nasa_power_data_filtered_within.xlsx'

# Read the Excel file using pandas
dataku = pd.read_excel(file_path, engine='openpyxl')

# Display the dataframe
print(dataku)
