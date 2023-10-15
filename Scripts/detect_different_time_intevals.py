import numpy as np

data = np.load('../Datasets/Sea Surface Height (SSH)/ssh_praticagem.npz')

dates = data["datetime_test"]

last_interval = 0
last_timestamp = dates[0]
different_intervals = []
index = 1
for timestamp in dates[1:]:
    current_interval = timestamp - last_timestamp
    if current_interval != last_interval:
        different_intervals.append(f"Intervalos de {(current_interval / np.timedelta64(1, 'm'))} minutos a partir da leitura {index}")
    last_timestamp = timestamp
    last_interval = current_interval
    index += 1

for element in different_intervals:
    print(element)
print(f"{len(different_intervals)} mudanças de intervalo.")