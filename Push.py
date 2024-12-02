from __future__ import print_function
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
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
                '/opt/airflow/credentials.json', SCOPES)      
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path, folder_id):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_name} with File ID: {file.get('id')}")

def download_file(service, file_id, output_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    with open(output_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

def main():
    # Authenticate and get the Google Drive service
    service = authenticate()

    # Folder ID where files will be uploaded in Google Drive
    folder_id = '1DYLjUQ86GNaAuTkCzcmJIc_4lkn_Tx1N'  

    # Files to upload
    files_to_upload = [
        'LiveData/BTC_1d.csv',
        'LiveData/cleaned_BTC_1d.csv',
        'Model/lstm_model.keras'
    ]

    # Upload each file
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            upload_file(service, file_path, folder_id)
        else:
            print(f"File not found: {file_path}")

    # Example: Download a file from Google Drive (Replace with your file ID)
    # download_file(service, 'your_file_id_here', 'output_path_here')

if __name__ == '__main__':
    main()
