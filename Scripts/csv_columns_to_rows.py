import csv

# Read the single-line CSV file
with open('../personal_files/datetime_train.csv', 'r') as single_line_csv:
    reader = csv.reader(single_line_csv)
    data = next(reader)  # Read the single line of values

# Create a new CSV file with each value in a separate row
with open('output_multi_row.csv', 'w', newline='') as multi_row_csv:
    writer = csv.writer(multi_row_csv)
    for value in data:
        writer.writerow([value])

print("Data has been converted to multi-row CSV.")
