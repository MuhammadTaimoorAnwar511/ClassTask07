from __future__ import print_function
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    """Authenticate and return the Google Drive service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_id, output_path):
    """Download a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        with open(output_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Downloading {output_path}: {int(status.progress() * 100)}% complete.")
        print(f"Downloaded {output_path} successfully.")
    except Exception as e:
        print(f"An error occurred while downloading {output_path}: {e}")

def main():
    # Authenticate and get the Google Drive service
    service = authenticate()

    # File IDs and corresponding output paths
    files_to_download = [
        {
            "file_id": "1aVmshx9_u03vIUFOMv1cbN8i9xQRFVQ3",  # File ID for BTC_1d.csv
            "output_path": "LiveData/BTC_1d.csv"
        },
        {
            "file_id": "1XXYqFKGrGyaCU0YMQv1LQLevhxlf7-t3",  # File ID for cleaned_BTC_1d.csv
            "output_path": "LiveData/cleaned_BTC_1d.csv"
        },
        {
            "file_id": "1_p4w5mjkgptd7HbJTjBmgoKBPhUvVbFl",  # File ID for lstm_model.keras
            "output_path": "Model/lstm_model.keras"
        }
    ]

    # Ensure the necessary directories exist
    os.makedirs("LiveData", exist_ok=True)
    os.makedirs("Model", exist_ok=True)

    # Download each file
    for file_info in files_to_download:
        download_file(service, file_info["file_id"], file_info["output_path"])

if __name__ == '__main__':
    main()
