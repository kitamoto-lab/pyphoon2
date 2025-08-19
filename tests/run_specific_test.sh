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

echo "Starting the test..."
# python -m unittest test_DigitalTyphoonDataset_MultiChannel.TestDigitalTyphoonDatasetMultiChannel
# python3 -m unittest test_DigitalTyphoonDataset_MultiChannel.TestDigitalTyphoonDatasetMultiChannel.test__initialize_and_populate_images_into_sequences

python3 -m unittest test_DigitalTyphoonDataset.TestDigitalTyphoonDataset.test_get_nonempty_seasons_and_sequences
# python3 -m unittest test_DigitalTyphoonImage.TestDigitalTyphoonImage.test_initialization_with_no_spectrum
# python3 -m unittest test_DigitalTyphoonSequence.TestDigitalTyphoonSequence
# python3 -m unittest test_DigitalTyphoonUtils.TestDigitalTyphoonUtils
