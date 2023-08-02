Example 1: Training a classifier based on image frames
======================================================


In this example we train a ResNet model using images loaded from the dataset. 

The model accepts single images, and is trained in batches of 16. The dataset is split into 80% train and 20% test, split by sequence so that all images in one sequence remain in the same train/test set as each other.


The Code
-----------

::

	import torch
	from torch import nn
	from torch import optim
	import torch.nn.functional as F
	import numpy as np
	import pandas as pd
	from tqdm import tqdm
	from torchvision import datasets, transforms, models
	import argparse
	from pathlib import Path
	from torch.utils.data import DataLoader

	from DigitalTyphoonDataloader.DigitalTyphoonDataset import DigitalTyphoonDataset

	def main(args):


	    ## Prepare the data

	    # Specify the paths to the data
	    data_path = args.dataroot
	    images_path = data_path + '/image/' # to the image folder
	    metadata_path = data_path + '/metadata/' # to the metadata folder
	    json_path = data_path + '/metadata.json'  # to the metadata json

	    # Define a filter to pass to the loader. 
	    #     Any image that the function returns true will be included
	    def image_filter(image):
	        return image.grade() < 7

	    # Define a function to transform each image, to pass to the loader.
	    # Crucially, this transform function is applied to each *image*, prior to any Pytorch processing.
	    # So, image-by-image transforms (i.e. clipping, downsampling, etc. can/should be done here)
	    def transform_func(image_ray):
	        # Clip the pixel values between 150 and 350
	        image_ray = np.clip(image_ray, standardize_range[0], standardize_range[1])

	        # Standardize the pixel values between 0 and 1
	        image_ray = (image_ray - standardize_range[0]) / (standardize_range[1] - standardize_range[0])

	        # Downsample the images to 224, 224
	        if downsample_size != (512, 512):
	            image_ray = torch.Tensor(image_ray)
	            image_ray = torch.reshape(image_ray, [1, 1, image_ray.size()[0], image_ray.size()[1]])
	            image_ray = nn.functional.interpolate(image_ray, size=downsample_size, mode='bilinear', align_corners=False)
	            image_ray = torch.reshape(image_ray, [image_ray.size()[2], image_ray.size()[3]])
	            image_ray = image_ray.numpy()
	        return image_ray

	    # Load Dataset
	    dataset = DigitalTyphoonDataset(str(images_path),
	                                    str(metadata_path),
	                                    str(json_path),
	                                    'grade',  # the labels we'd like to retrieve from the dataset
	                                    filter_func=image_filter, # the filter function defined above
	                                    transform_func=transform_func, # the transform function defined above
	                                    verbose=False)

	    # Split the dataset into a training and test split (80% and 20% respectively)
	    #   split by sequence so all images in one sequence will belong to the same bucket 
	    train_set, test_set = dataset.random_split([0.8, 0.2], split_by='sequence')

	    # Make Pytorch DataLoaders out of the returned sets. From here, it retains all Pytorch functionality.
	    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


	    ## Prepare the model

	    # Hyperparameters
	    num_epochs = args.max_epochs
	    batch_size = 16
	    learning_rate = 0.001
	    standardize_range = (150, 350)
	    downsample_size = (224, 224)

	    # Load a ResNet model 
	    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=weights)
	    # Modify the model to take single channel images
	    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	    # Modify the model to classify between 7 classes
	    model.fc = nn.Linear(in_features=512, out_features=7, bias=True)

	    # Loss and optimizer
	    criterion = nn.CrossEntropyLoss()
	    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


	    ## Train the model
	    for epoch in np.arange(max_epochs):

	        batches_per_epoch = len(trainloader)

	        model.train()

	        for batch_num, data in enumerate(tqdm(trainloader)):
	            # One batch of the data (16 images and 16 labels) are held in the data variable

	            # Data is a tuple, with images in data[0] and labels in data[1]
	            images, labels = data

	            # cast pixels to float and grade (label) to long
	            images, labels = torch.Tensor(images).float(), torch.Tensor(labels).long()

	            # Reshape the image tensor to add a channel dimension (only one channel)
	            images = torch.reshape(images, [images.size()[0], 1, images.size()[1], images.size()[2]])

	            optimizer.zero_grad()

	            # Forward pass
	            predictions = model(images)

	            # Calculate the loss
	            loss = criterion(predictions, labels)
	        
	            # backward pass
	            loss.backward()
	            # update weights
	            optimizer.step()


	if __name__ == '__main__':
	    parser = argparse.ArgumentParser(description='Train a resnet model')
	    parser.add_argument('--dataroot', required=True, type=str, help='path to the root data directory')
	    parser.add_argument('--split_by', default='frame', type=str, help='How to split the dataset')
	    parser.add_argument('--maxepochs', default=100, type=int)
	    args = parser.parse_args()

	    main(args)


