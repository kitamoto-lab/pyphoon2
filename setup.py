from setuptools import setup, find_packages

setup(
    name='DigitalTyphoonDataloader',
    version='1.0.0',
    description='Dataloader for the Kitamoto Lab Digital Typhoon Dataset',
    url='https://github.com/jared-hwang/DigitalTyphoonDataset',
    author='Jared Hwang',
    author_email='jared.hwang@gmail.com',
    # license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['mpi4py>=2.0',
                      'numpy',
                      'torch',
                      'torchvision',
                      'pandas',
                      'h5py'
                      ],
    zip_safe=False
)
