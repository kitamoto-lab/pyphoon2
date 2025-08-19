# Computation parameters
IMAGE_DIR = 'test_data_files/image/'
METADATA_DIR = 'test_data_files/metadata/'
METADATA_JSON = 'test_data_files/metadata.json'


IMAGE_DIRS = ['/dataset/2/wnp/image/',
              '/dataset/1/wnp/image/']
METADATA_DIRS = ['/dataset/2/wnp/metadata/',
                 '/dataset/1/wnp/metadata/']
METADATA_JSONS = ['/dataset/2/wnp/metadata.json',
                  '/dataset/1/wnp/metadata.json']

# Print for debugging
import os
print("Current working directory:", os.getcwd())

# Constants defined in this file
print("IMAGE_DIR:", IMAGE_DIR)
print("METADATA_DIR:", METADATA_DIR)
print("METADATA_JSON:", METADATA_JSON)

# Check if directories exist
print("IMAGE_DIR exists:", os.path.exists(IMAGE_DIR))
print("METADATA_DIR exists:", os.path.exists(METADATA_DIR))
print("METADATA_JSON exists:", os.path.exists(METADATA_JSON))
