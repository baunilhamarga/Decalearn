import numpy as np

data = np.load('../Datasets/Sea Surface Height (SSH)/ssh_praticagem.npz')

dates = data["datetime_train"]

# Get the indices that would sort the array
sorted_indices = np.argsort(dates)

# Use the sorted indices to reorder the array
dates = dates[sorted_indices]

last_interval = 0
last_timestamp = dates[0]
different_intervals = []
frequencies = {}
index = 1
intervals_10 = []
ten_count = 0

for timestamp in dates[1:]:
    current_interval = timestamp - last_timestamp
    minutes_delta = (current_interval / np.timedelta64(1, 'm'))

    if current_interval != last_interval:
        interval_description = f"Intervalos de {minutes_delta} minutos a partir da leitura {index}"
        different_intervals.append(interval_description)
        frequencies.setdefault(int(minutes_delta), 0)
        if int(minutes_delta) == 10:
            intervals_10.append(ten_count)
            ten_count = 0

    if int(minutes_delta) == 10:
        ten_count += 1    


    frequencies[minutes_delta] += 1
    last_timestamp = timestamp
    last_interval = current_interval
    index += 1

for element in different_intervals:
    print(element)

# Sort the frequencies dictionary in descending order
sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[1], reverse=True))

print(f"{len(different_intervals)} mudanças de intervalo.")
print(sorted_frequencies)
print(sum(frequencies.values()))
print(dates.shape)
print(len(intervals_10))