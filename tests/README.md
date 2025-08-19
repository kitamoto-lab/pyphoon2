# Tests for Digital Typhoon Project

This folder contains unit tests for the Digital Typhoon project. Follow the instructions below to set up the testing environment and run the tests.

---

## **Setup Testing Data**

To run the tests, you need testing data. The scripts provided will automatically check if the required data is present, and if not, they will download and extract it for you.

1. **Manual Setup (Optional)**:
   If you prefer manual setup, download the testing data:
   ```
   https://minio.hisoft.com.vn/anhtn/test_data_files.zip
   ```

   Unzip the data:
   ```bash
   unzip test_data_files.zip -d test_data_files
   ```

2. **Automated Setup**:
   The scripts `run_all_tests.sh` and `run_specific_test.sh` automatically ensure the required testing data is downloaded and extracted before running tests.

---

## **Contents of the `tests` Folder**

This folder contains the following test files:

- `test_DigitalTyphoonDataset.py`: Unit tests for the `DigitalTyphoonDataset` module.
- `test_DigitalTyphoonImage.py`: Unit tests for the `DigitalTyphoonImage` module.
- `test_DigitalTyphoonSequence.py`: Unit tests for the `DigitalTyphoonSequence` module.
- `test_DigitalTyphoonUtils.py`: Unit tests for utility functions used in the Digital Typhoon project.

---

## **How to Run the Tests**

### **Run All Tests**
Use the `run_all_tests.sh` script to execute all tests in the folder. The script will automatically check if the test data is present. If not, it will download and extract the required data before running the tests:
```bash
./run_all_tests.sh
```

### **Run a Specific Test**
To run a specific test file, use the `run_specific_test.sh` script. Similar to the `run_all_tests.sh` script, this will ensure the test data is downloaded and set up before running the specified test:
```bash
./run_specific_test.sh test_DigitalTyphoonSequence.py
```

Alternatively, you can manually specify the test file using Python's `unittest` module:
```bash
python3 -m unittest test_DigitalTyphoonSequence
```

---

## **Scripts**

- **`run_all_tests.sh`**: A script to run all tests in the folder. This script checks if the required test data is present and downloads it if necessary.
- **`run_specific_test.sh`**: A script to run a specific test file. It also ensures that the required test data is downloaded before running the test.

---

## **Dependencies**

Ensure you have the following installed in your Python environment:
- `Python 3.10` or later
- `unittest` module (bundled with Python)
- `wget` and `unzip` commands installed on your system for downloading and extracting test data.

You can install additional dependencies if required using `pip`:
```bash
pip install -r requirements.txt
```

---

## **Folder Structure**

After extracting the testing data (either manually or automatically), your folder structure should look like this:
```
tests/
├── run_all_tests.sh
├── run_specific_test.sh
├── test_DigitalTyphoonDataset.py
├── test_DigitalTyphoonImage.py
├── test_DigitalTyphoonSequence.py
├── test_DigitalTyphoonUtils.py
└── test_data_files/
    ├── metadata/
    ├── image/
    │   ├── 200801/
    │   └── ...
    └── ...
```

---

Feel free to reach out for support or questions related to this project!
