import os
import gzip

directory = "results_per_section"
dest_dir = "csv_data_files"

os.makedirs(dest_dir, exist_ok=True)

# List to store .gz files for extraction
gz_files = [f for f in os.listdir(directory) if f.endswith('.gz')]

# Convert each .gz file to .csv
for filename in os.listdir(source_dir):
    if filename.endswith(".gz"):
        gz_path = os.path.join(source_dir, filename)
        csv_filename = filename[:-3]  # Remove .gz extension
        csv_path = os.path.join(dest_dir, csv_filename)

        # Read and write the decompressed file
        with gzip.open(gz_path, 'rt') as f_in, open(csv_path, 'w') as f_out:
            f_out.write(f_in.read())

print("Conversion complete. CSV files saved to:", dest_dir)
