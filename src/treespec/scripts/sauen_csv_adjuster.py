"Script to convert csv file from Sauen data to required format of the container."
import pandas as pd #type: ignore

# --- SETTINGS ---
input_file = "input.csv"   # path to your CSV
output_file = "output.csv" # path for modified CSV
column_name = "angle"      # column to adjust
rotation = 180             # degrees to add

# Read CSV
df = pd.read_csv(input_file)

# Adjust the specified column by 180 degrees
# Wrap values back into the range [0, 360)
df[column_name] = (df[column_name] + rotation) % 360

# Save the modified CSV
df.to_csv(output_file, index=False)

print(f"Updated CSV saved as '{output_file}'")
