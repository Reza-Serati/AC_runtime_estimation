import os
import pandas as pd

INPUT_FILE = "data/all_sensor_data.csv"
OUTPUT_DIR = "data/by_room"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for chunk in pd.read_csv(INPUT_FILE, chunksize=500_000):
    for sensor_id, group in chunk.groupby("sensor.number"):
        output_file = os.path.join(OUTPUT_DIR, f"sensor_{sensor_id}.csv")
        file_exists = os.path.exists(output_file)

        group.to_csv(
            output_file,
            mode="a",
            index=False,
            header=not file_exists
        )

print("Done. Files saved in:", OUTPUT_DIR)