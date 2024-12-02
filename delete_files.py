import os

def delete_file(file_path):
    """
    Deletes the file at the given path if it exists.
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    else:
        print(f"File does not exist: {file_path}")

def main():
    # Specify the files to be deleted
    files_to_delete = [
        "LiveData/cleaned_BTC_1d.csv",
        "Model/lstm_model.keras"
    ]
    
    # Delete each file if it exists
    for file_path in files_to_delete:
        delete_file(file_path)

if __name__ == "__main__":
    main()
