import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mindspore as ms
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mindspore import nn, Tensor
import mindspore.dataset as ds
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, Callback


df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

df['temp'] = df['temp'] - 273.15

df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')

df["hour"] = df["date_time"].dt.hour  
df["day_of_week"] = df["date_time"].dt.dayofweek  

df.drop(columns=['holiday'], inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def add_past_traffic_features(data, past_steps=6):
    for i in range(1, past_steps + 1):
        data[f'past_traffic_{i}'] = data['traffic_volume'].shift(i)
    data.dropna(inplace=True)  # İlk birkaç satır eksik olacak, onları kaldır
    return data

df = add_past_traffic_features(df)

y = df['traffic_volume'].values
X = df.drop(columns=['traffic_volume', 'date_time']).values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)
y = scaler.fit_transform(y)

def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  
X_seq, y_seq = create_sequences(X, y, time_steps)

train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# -----------------------------------
# Model Parametreleri
# -----------------------------------

input_size = X_train.shape[2]
hidden_size = 256
output_size = 1
num_layers = 4
dropout_rate = 0.3
epochs = 25  

# -----------------------------------
# LSTM Modeli
# -----------------------------------

class ImprovedLSTMModel(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(ImprovedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Dense(hidden_size, output_size)

    def construct(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# -----------------------------------
# Eğitim için Dataset Hazırlığı
# -----------------------------------
batch_size = 128
train_data = ds.NumpySlicesDataset({"array": X_train, "label": y_train}, shuffle=True).batch(batch_size)
test_data = ds.NumpySlicesDataset({"array": X_test, "label": y_test}, shuffle=False).batch(batch_size)

# -----------------------------------
# Öğrenme Oranı (Cosine Annealing LR)
# -----------------------------------
steps_per_epoch = len(train_data) 
decay_epoch = epochs  

lr_scheduler = nn.cosine_decay_lr(min_lr=1e-5, max_lr=0.0003, total_step=epochs * steps_per_epoch,
                                  step_per_epoch=steps_per_epoch, decay_epoch=decay_epoch)

# -----------------------------------
# Model Tanımlama
# -----------------------------------
model = ImprovedLSTMModel(input_size, hidden_size, output_size, num_layers)
loss_fn = nn.SmoothL1Loss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=lr_scheduler)
train_net = Model(model, loss_fn=loss_fn, optimizer=optimizer)

# -----------------------------------
# Modeli Eğitme
# -----------------------------------

train_net.train(epochs, train_data, callbacks=[LossMonitor()])


# -----------------------------------
# Modeli Değerlendirme
# -----------------------------------


y_pred = []
for batch in test_data.create_dict_iterator():
    X_batch = batch["array"]
    y_batch_pred = model(Tensor(X_batch, ms.float32))
    y_pred.extend(y_batch_pred.asnumpy())

# Ölçeklendirilmiş değerlere geri dön
y_test_original = scaler.inverse_transform(y_test)
y_pred_original = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))

# Hata metriklerini hesapla
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Scatter plot: Gerçek vs. Tahmin edilen değerler
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         color='red', linestyle='--', linewidth=2)  
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Traffic Volume")
plt.show()

