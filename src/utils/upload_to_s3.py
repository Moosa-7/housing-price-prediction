import boto3
import os
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

# ==========================================
# 1. LOAD CREDENTIALS (DEBUG MODE)
# ==========================================
# We explicitly tell Python: "Look for .env in the current folder"
load_dotenv()

print("🔍 DEBUG: Checking credentials...")
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

if not access_key:
    print("❌ ERROR: .env file was found, but AWS_ACCESS_KEY_ID is empty.")
    print("👉 Check your .env file. It should look like: AWS_ACCESS_KEY_ID=AKIA...")
    print("👉 Make sure you SAVED the .env file (Ctrl+S).")
    exit(1)
elif " " in access_key:
    print("❌ ERROR: Your Access Key has spaces in it!")
    print("👉 Remove any spaces around the '=' sign in your .env file.")
    exit(1)
else:
    print(f"✅ Credentials loaded! Key starts with: {access_key[:4]}...")

# ==========================================
# 2. CONFIGURATION
# ==========================================
BUCKET_NAME = "housing-project-moosa-2025"  # Ensure this is unique
REGION = "eu-west-2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def create_bucket_if_not_exists(s3_client, bucket_name, region=None):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists.")
    except ClientError:
        print(f"⚠️ Bucket '{bucket_name}' not found. Creating it...")
        try:
            if region is None:
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                location = {'LocationConstraint': region}
                s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
            print(f"✅ Created bucket '{bucket_name}' successfully.")
        except ClientError as e:
            print(f"❌ Failed to create bucket: {e}")
            exit(1)

def upload_files(s3_client, bucket_name, local_path):
    if not local_path.exists():
        print(f"❌ Data directory not found: {local_path}")
        return

    print(f"\n🚀 Starting upload from {local_path}...")
    files_uploaded = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = Path(root) / file
            s3_key = f"data/processed/{file}"
            try:
                s3_client.upload_file(str(local_file_path), bucket_name, s3_key)
                print(f"   📤 Uploaded {file}")
                files_uploaded += 1
            except Exception as e:
                print(f"❌ Error uploading {file}: {e}")

    if files_uploaded > 0:
        print(f"\n✅ Success! Uploaded {files_uploaded} files to S3.")
    else:
        print("\n⚠️ No files were uploaded.")

if __name__ == "__main__":
    s3 = boto3.client('s3', region_name=REGION)
    create_bucket_if_not_exists(s3, BUCKET_NAME, REGION)
    upload_files(s3, BUCKET_NAME, DATA_DIR)