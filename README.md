# Metrics-Distance-TensorFlow

This repository contains Python scripts to calculate various distance metrics between rows of data in a dataset using TensorFlow. The implemented distance metrics include Dynamic Time Warping (DTW) Distance and Euclidean Distance.

## File List

1. **Import-Data-Drive.py**: This script is used to import data from Google Drive to Google Colab using Google Colab.

2. **dtw-calculate.py**: This script calculates the Dynamic Time Warping (DTW) Distance between each pair of data rows in the dataset using TensorFlow.

3. **dtw-save-file.py**: This script saves the DTW distance matrix to a CSV file.

4. **dtw-view-result.py**: This script loads and displays the calculated DTW distance results that have been saved in CSV format.

5. **euclidean-calculate.py**: This script calculates the Euclidean distance between each pair of data rows in the dataset using TensorFlow.

6. **euclidean-save-file.py**: This script saves the Euclidean distance matrix to a CSV file.

7. **euclidean-view-result.py**: This script loads and displays the calculated Euclidean distance results that have been saved in CSV format.

## How to Use

1. Ensure you have Google Colab or a Python environment with the required dependencies installed.
2. Make sure the data to be processed is available in your Google Drive, or adjust the file paths used in the scripts accordingly.
3. Run the appropriate scripts to import data, calculate distances, save results, and view the results as needed.

## Important Notes

1. Make sure you have installed the required dependencies before running the scripts, such as TensorFlow and fastdtw.
2. Note that some scripts may require adjustments to file paths according to the location of your data in Google Drive.
3. Be sure to read the documentation and comments within each script before using or modifying them.

The data file [nasa_power_data_filtered_within.xlsx](https://github.com/zenklinov/nasapower/blob/main/nasa_power_data_filtered_within.xlsx) located in the repository [zenklinov/nasapower](https://github.com/zenklinov/nasapower) is the dataset mentioned above, which is stored in my drive.

Please refer to the README.md and individual scripts for more information on how to use and modify the scripts in this repository.
