# Your-Smart-Home-Can-t-Keep-a-Secret-Towards-Automated-Fingerprinting-of-IoT-Traffic-with-Neural-Net
A implementation of Your Smart Home Can’t Keep a Secret: Towards Automated Fingerprinting of IoT Traffic with Neural Networks

## Run the code

1. Download the dataset from https://drive.google.com/drive/folders/1aQXlBKZaal8MgsrYx2hyanorwMZEKv6e and put it in the  *\dataset\processing_raw_data\data* folder
2. run pcap2csv.py,  extract_features.ipynb, Dataset.py to prepare the dataset
3. run demo/lstm.py to derive the result

![result](C:\Users\ms396\Documents\GitHub\Your-Smart-Home-Can-t-Keep-a-Secret-Towards-Automated-Fingerprinting-of-IoT-Traffic-with-Neural-Net\result.PNG)

## Remain to do

1. Note that since the authors did not publish the noisy data,  I haven't implement the *Dataset-Noise* version 
2. With this framework, Random Forest& Bi-LSTM  is easy
3. Modifying the Dataset.py to try different window size

