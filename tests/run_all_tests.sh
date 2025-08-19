#!/bin/bash

# Variables
ZIP_FILE="test_data_files.zip"
DOWNLOAD_URL="https://minio.hisoft.com.vn/anhtn/test_data_files.zip"
TARGET_DIR="test_data_files/image/200801"
EXTRACT_DIR="./"

# Function to download the ZIP file
download_zip() {
  echo "Downloading $ZIP_FILE from $DOWNLOAD_URL..."
  wget -O "$ZIP_FILE" "$DOWNLOAD_URL"
  
  if [ $? -ne 0 ]; then
    echo "Error: Failed to download $ZIP_FILE. Exiting."
    exit 1
  fi
  echo "Download completed successfully."
}

# Function to unzip the ZIP file
unzip_files() {
  echo "Extracting $ZIP_FILE to $EXTRACT_DIR/..."
  unzip "$ZIP_FILE" -d "$EXTRACT_DIR"
  
  if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip $ZIP_FILE. Exiting."
    exit 1
  fi
  echo "Extraction completed successfully."
}

# Main script execution
if [ -d "$TARGET_DIR" ]; then
  echo "Directory '$TARGET_DIR' already exists. Skipping download and extraction."
else
  if [ -f "$ZIP_FILE" ]; then
    echo "ZIP file '$ZIP_FILE' already exists. Skipping download."
  else
    download_zip
  fi

  # Only unzip if the target directory does not exist
  if [ ! -d "$TARGET_DIR" ]; then
    unzip_files
  else
    echo "After checking, directory '$TARGET_DIR' exists. Skipping extraction."
  fi

  # Optional: Remove the ZIP file after extraction
  # rm "$ZIP_FILE"
fi

# Run all Python tests in the current directory
echo "Running Python unittests..."
python3 -m unittest discover -s . -p "*.py"

# Check if tests ran successfully
if [ $? -ne 0 ]; then
  echo "Some tests failed."
  exit 1
else
  echo "All tests passed successfully."
fi
