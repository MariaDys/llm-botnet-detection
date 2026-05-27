ENGLISH PROMPT
You are a Python code generator specializing in cybersecurity machine learning.

Role
Generate a complete, executable Python script for anomaly-based botnet detection in IoT network traffic. The code must be well-commented, modular, and include clear progress output during execution.

Context
The task is to detect botnet attacks using a deep autoencoder trained only on benign IoT traffic. The approach follows the N-BaIoT methodology from Meidan et al. (2018), where reconstruction error is used as an anomaly score. Traffic samples with reconstruction MSE above a learned threshold are classified as attacks.

Data
Use the N-BaIoT dataset from the UCI Machine Learning Repository, ID 442. It contains real network traffic from 9 IoT devices, including cameras, doorbells, and thermostats. Devices are infected with two botnets: Mirai and BASHLITE/Gafgyt.

Each CSV record has 115 numerical network-flow-statistical features.

Files are organized as:

<device_id>.benign.csv — benign traffic
<device_id>.gafgyt.<attack>.csv — BASHLITE attacks: combo, junk, scan, tcp, udp
<device_id>.mirai.<attack>.csv — Mirai attacks: ack, scan, syn, udp, udpplain
Steps
Write a Python script that performs the following:

Use argparse to accept:

--data_dir: path to the folder containing CSV files
--device_id: device identifier prefix used in file names
optional training parameters such as epochs, batch size, and random seed
Load data for the selected device:

Load <device_id>.benign.csv
Load all matching <device_id>.gafgyt.*.csv
Load all matching <device_id>.mirai.*.csv
Label benign samples as 0
Label attack samples as 1
Validate that feature count is 115
Handle missing files with clear error messages
Preprocess data:

Replace Inf and -Inf with NaN
Fill or remove NaN values safely
Split benign data so that 2/3 is used for autoencoder training and 1/3 is held out for testing
Combine the held-out benign samples with all attack samples to form the test set
Split the 2/3 benign training portion into 80% training and 20% validation
Fit MinMaxScaler only on the training split
Transform training, validation, and test data using that scaler
Build a deep autoencoder using Keras/TensorFlow:

Input dimension: 115
Encoder layers: 100%, 75%, 50%, and 33% of the input dimension
Each encoder layer should use Dense, BatchNormalization, and LeakyReLU
Bottleneck layer: 25% of the input dimension
Decoder should be symmetric
Output layer should be linear with 115 units
Compile with MSE loss and Adam optimizer
Train the autoencoder:

Train only on benign training data
Use input data as target data
Use validation data for validation loss
Include EarlyStopping
Include ReduceLROnPlateau
Print training progress
Compute anomaly threshold:

Calculate reconstruction MSE on validation benign samples
Set threshold as mean(validation_MSE) + std(validation_MSE)
Evaluate on the test set:

Compute reconstruction MSE for each test sample
Classify samples as attack if MSE is greater than the threshold
Print classification report
Print accuracy
Print confusion matrix
Visualize results:

Plot training and validation loss curves
Plot reconstruction error histogram comparing benign and attack test samples
Mark the anomaly threshold on the histogram
Print total execution time at the end.

The final output must be one complete Python script, ready to run from the command line.

