## Digital Typhoon Dataset Dataloader

<!-- ABOUT THE PROJECT -->
## About The Project

The Digital Typhoon Project is a project aimed to be an example of the application of meteoinformatics to large-scale 
real-world issues. The two primary challenges of this project are to (1) build, for the typhoon image collection, 
large-scale scientific databases which are the foundation of meteoinformatics, and (2) to establish algorithms 
and database models for the discovery of information and knowledge useful for typhoon analysis and prediction. 

The Dataloader contained within this project addresses the first challenge. It is built for the Kitamoto Lab typhoon 
dataset and its structure, and is designed to be a easily accessible PyTorch-based interface with the dataset. Through 
it the user can (1) access typhoon images via index, typhoon ID, or year, (2) load all data into memory if desired, and
(3) randomly split the dataset into buckets for model training, by image, year, or typhoon ID while preventing leakage 
between the buckets. 


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

_Below are the libraries required to use the loader. Python3 is required. Installation instructions are below._

* PyTorch
  ```sh
  pip3 install torch torchvision  
  ```
  
* numpy
  ```sh
  pip3 install numpy
  ```
* pandas
  ```sh
  pip3 install pandas  
  ```
* h5
  ```sh
  pip3 install h5py
  ```
  
### Usage

_Below is a brief example on how to initialize and access data using the DataLoader:_ 

1. Clone the repo
    ```sh
    git clone https://github.com/jared-hwang/DigitalTyphoonDataset.git
    ```
2. Import the Dataloader class
    ```python
    from DigitalTyphoonDataset import DigitalTyphoonDataset
    ```
3. Instantiate the loader
    ```python
    # See the documentation for description of the optional parameters. 
    dataset_obj = DigitalTyphoonDataset("/path/to/image/directory", 
                                        "/path/to/track/directory", 
                                        "/path/to/metadata.json", 
                                        split_dataset_by='sequence',
                                        load_data_into_memory=False,
                                        verbose=True)
    ```
The dataset object is now instantiated and you can use the data in the desired fashion. Some examples include: 

* Get the length of the dataset
    ```python
    length = len(dataset_obj)
    ```
  
* Get the item at the i'th index
    ```python
    image = dataset_obj[i]    
    ```  
  
* Split the dataset into train, test, and validation sets
    ```python
    train, test, val = dataset_obj.random_split([0.7, 0.15, 0.15], split_by='sequence')
    ```
