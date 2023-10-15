import numpy as np
import csv

if __name__ == '__main__':
    # Replace 'your_data.npz' with the actual path to your .npz file.
    npz_file = np.load('activity_recognition/ssh_praticagem.npz')

    # Loop through the keys in the .npz file and save each array as a separate CSV file.
    for key in npz_file.keys():
        data_array = npz_file[key]

        # Create a CSV filename based on the key.
        csv_filename = f'{key}.csv'

        # Open a CSV file for writing.
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Flatten the matrix and write it to the CSV file.
            csv_writer.writerow([str(value) for value in data_array])

        print(f'{key} has been saved to {csv_filename}')
