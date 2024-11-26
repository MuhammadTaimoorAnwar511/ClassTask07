import os
import pandas as pd

# Folder containing CSV files
folder_path = "LiveData"

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure only CSV files are processed
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Drop the 6th column (first "Volume") if it exists
            if len(df.columns) > 5:  # Ensure the 6th column exists
                df.drop(columns=df.columns[5], inplace=True)
                print(f"6th column (first 'Volume') removed from {file_name}")

            # Rename "Volume.1" to "Volume" if it exists
            if "Volume.1" in df.columns:
                df.rename(columns={"Volume.1": "Volume"}, inplace=True)
                print(f"'Volume.1' renamed to 'Volume' in {file_name}")

            # Check for duplicate rows
            duplicates = df[df.duplicated(keep=False)]

            if not duplicates.empty:
                # Get indices of duplicate rows
                duplicate_indices = duplicates.index.tolist()

                # Identify consecutive duplicate groups
                drop_indices = []
                for i in range(len(duplicate_indices) - 1):
                    # If consecutive indices are the same, mark for deletion
                    if duplicate_indices[i] + 1 == duplicate_indices[i + 1]:
                        drop_indices.append(duplicate_indices[i])

                # Drop marked rows
                df = df.drop(index=drop_indices)
                print(f"Consecutive duplicates removed from {file_name}")

            # Save the cleaned DataFrame with a new name
            new_file_name = f"cleaned_{file_name}"
            new_file_path = os.path.join(folder_path, new_file_name)
            df.to_csv(new_file_path, index=False)
            print(f"Cleaned file saved as: {new_file_name}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
