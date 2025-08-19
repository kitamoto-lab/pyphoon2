from setuptools import setup, find_packages

setup(
    name='pyphoon2',
    version='2.0.0',
    description='Dataloader for the Kitamoto Lab Digital Typhoon Dataset',
    url='https://github.com/tonganh/pyphoon2',
    author='Jared Hwang',
    author_email='kitamoto@nii.ac.jp',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
                      'numpy',
                      'torch',
                      'torchvision',
                      'pandas',
                      'h5py',
                      ],
    zip_safe=False
)
