# Traffic Volume Prediction Using LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) model using MindSpore to predict traffic volume based on historical data and various weather conditions. The dataset contains traffic volume measurements along with weather parameters, and the model is trained to forecast future traffic trends.

## Dataset
The dataset used is `Metro_Interstate_Traffic_Volume.csv`, which includes the following columns:
- `holiday`: Whether the day is a holiday (removed during preprocessing)
- `temp`: Temperature in Kelvin (converted to Celsius)
- `rain_1h`: Rain volume in the last hour
- `snow_1h`: Snow volume in the last hour
- `clouds_all`: Cloud coverage percentage
- `weather_main`: General weather condition
- `weather_description`: Detailed weather description
- `date_time`: Timestamp of the record
- `traffic_volume`: Traffic volume at that time

### Sample Data
| temp (°C) | rain_1h | snow_1h | clouds_all | weather_main | weather_description | date_time           | traffic_volume |
|-----------|---------|---------|------------|--------------|----------------------|---------------------|----------------|
| 15.13     | 0.0     | 0.0     | 40         | Clouds       | scattered clouds    | 2012-10-02 09:00:00 | 5545           |
| 16.21     | 0.0     | 0.0     | 75         | Clouds       | broken clouds       | 2012-10-02 10:00:00 | 4516           |
| 16.43     | 0.0     | 0.0     | 90         | Clouds       | overcast clouds     | 2012-10-02 11:00:00 | 4767           |
| 16.98     | 0.0     | 0.0     | 90         | Clouds       | overcast clouds     | 2012-10-02 12:00:00 | 5026           |
| 18.01     | 0.0     | 0.0     | 75         | Clouds       | broken clouds       | 2012-10-02 13:00:00 | 4918           |
| 18.59     | 0.0     | 0.0     | 1          | Clear        | sky is clear        | 2012-10-02 14:00:00 | 5181           |
| 20.04     | 0.0     | 0.0     | 1          | Clear        | sky is clear        | 2012-10-02 15:00:00 | 5584           |

## Preprocessing
1. Convert temperature from Kelvin to Celsius.
2. Convert `date_time` to `datetime` format.
3. Extract hour and day of the week.
4. Remove `holiday` column.
5. One-hot encode categorical features (`weather_main`, `weather_description`).
6. Add past traffic data as features (last 6 steps).
7. Scale features using MinMaxScaler.

## Model Implementation
The LSTM model is built with:
- **Input Size**: Number of features after preprocessing
- **Hidden Size**: 256
- **Number of Layers**: 4
- **Dropout Rate**: 0.3
- **Output Size**: 1
- **Batch Size**: 128
- **Training Epochs**: 25
- **Optimizer**: Adam
- **Loss Function**: SmoothL1Loss
- **Learning Rate Scheduler**: Cosine annealing

## Training
- The dataset is split into **80% training** and **20% testing**.
- Sequential data is structured into time steps of **24 hours**.
- The model is trained using a dataset created with MindSpore's `NumpySlicesDataset`.
- Training is monitored with `LossMonitor()`.

## Evaluation
After training, the model's predictions are compared to actual values using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

A scatter plot is generated to visualize actual vs. predicted traffic volume.

### Evaluation Results
```
Mean Absolute Error (MAE): 307.25
Mean Squared Error (MSE): 225081.80
R² Score: 0.94
```

### Actual vs. Predicted Scatter Plot
![65](https://github.com/user-attachments/assets/50e1282e-4e74-4bb3-a815-7b05d6c8c308)


## Execution
To run the script, ensure you have the dataset and dependencies installed:
```bash
pip install numpy pandas seaborn matplotlib mindspore scikit-learn
```
Then, execute the script:
```bash
python traffic_lstm.py
```
### Expected Output Example
```
Mean Absolute Error (MAE): 421.65
Mean Squared Error (MSE): 285621.78
R² Score: 0.76
```
A scatter plot will be displayed to compare actual vs. predicted values.

## Conclusion
This project demonstrates how LSTM networks can be used to predict traffic volume based on historical patterns and weather conditions. The model effectively captures trends and dependencies in the data, providing reliable predictions for future traffic volume.
