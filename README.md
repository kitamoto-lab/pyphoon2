## pyphoon2

<!-- ABOUT THE PROJECT -->
## About The Project

The Digital Typhoon Project is a project aimed to be an example of the application of meteoinformatics to large-scale 
real-world issues. The two primary challenges of this project are to (1) build, for the typhoon image collection, 
large-scale scientific databases which are the foundation of meteoinformatics, and (2) to establish algorithms 
and database models for the discovery of information and knowledge useful for typhoon analysis and prediction. 

The Dataloader contained within this project addresses the first challenge. It is built for the Kitamoto Lab typhoon 
dataset and its structure, and is designed to be a easily accessible PyTorch-based interface with the dataset. Through 
it the user can (1) access typhoon images via index, typhoon ID, or season, (2) load all data into memory if desired, and
(3) randomly split the dataset into buckets for model training, by image, season, or typhoon ID while preventing leakage 
between the buckets. 

See readthedocs documentation [here](https://pyphoon2.readthedocs.io/en/latest/).

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This project uses:
* python3
* torch
* torchvision
* numpy
* pandas
* h5py

### Installation

1. Clone and enter the repo 
    ```sh
    git clone https://github.com/kitamoto-lab/pyphoon2
    cd pyphoon2
    ```
2. Install the package
    ```sh
    pip3 install .
    ```
3. To uninstall, run
    ```sh
    pip3 uninstall pyphoon2
    ```
  
### Usage

_Below is a brief example on how to initialize and access data using the DataLoader:_ 

1. Import the Dataset class
    ```python
    from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
    ```
   You can also import the submodules `DigitalTyphoonSequence`, `DigitalTyphoonImage`, and 
    `DigitalTyphoonUtils` in the same way if desired:
    ```python
    from pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
    from pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
    from pyphoon2.DigitalTyphoonUtils import *
    ```
2. Instantiate the loader
    ```python
    # See the documentation for description of the optional parameters. 
    dataset_obj = DigitalTyphoonDataset("/path/to/image/directory/", 
                                        "/path/to/metadata/directory/", 
                                        "/path/to/metadata.json", 
                                        ('grade', 'lat', 'long'), # labels to return when indexing 
                                        split_dataset_by='sequence',
                                        load_data_into_memory=False,
                                        ignore_list=[],
                                        verbose=True)
    ```
The dataset object is now instantiated and you can use the data in the desired fashion. Some examples include: 

* Get the length of the dataset
    ```python
    length = len(dataset_obj)
    ```
  
* Get the item at the i'th index
    ```python
    image_array, labels = dataset_obj[i] # labels will correspond to labels passed in on instantiation or set via dataset.set_labels()
    image_obj = dataset_obj.get_image_at_idx(i)    
    image_obj.image() # Get the image pixels in a numpy array
    image_obj.year()  # Get the year the image was taken  
    image_obj.grade() # Get the grade of the typhoon at the time of the image
                  # For a full list of accessible image metadata, see documentation
    ```  
  
* Split the dataset into train, test, and validation sets
    ```python
    train, test, val = dataset_obj.random_split([0.7, 0.15, 0.15], split_by='sequence')
  
    trainloader = Dataloader(train, batch_size=16, shuffle=True)
    ```
