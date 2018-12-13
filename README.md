# Deep Active Learning for Myelin Segmentation on Histology Data

Open-source Active Learning simulation framework for segmenting myelin from histology data based on uncertainty sampling. Written in Python. Using the Keras framework. Based on a convolutional neural network architecture. Pixels are classified either as myelin or background.

![alt tag](https://github.com/neuropoly/deep_active_learning/blob/master/docs/activelearning_fig0.png)

## Installation

The following lines will help you install all you need to ensure that the Notebooks are working. Test data, instructions and results examples are provided to help you use this framework.

#### Python
First, you should make sure that Python 2.7 is installed on your computer. 
Run the following command in the terminal:
```
python -V
```
If you have the Anaconda distribution installed on your system, you can specify the version of Python that you want installed in your virtual environment set up below, even if it differs from the version displayed by the “python -V” command. To see the list of Python versions available to be installed in your conda virtual environment, run:
```
conda search python
```
#### Virtual Environment

We recommand you to set up a virtual environment. A virtual environment is a tool that lets you install specific versions of the python modules you want. It will allow to run this code with respect to its module requirements, without affecting the rest of your python installation.

If you have the Anaconda Distribution installed on your system, you can  use the conda virtual environment manager, which allows you to specify a different Python version to be installed in your virtual environment than what’s available by default on your system.

To create a virtual environment called “dal_venv” with the Anaconda Distribution, run:
```
conda create -n dal_venv python=2.7
```
To activate it, run the following command:

```
source activate dal_venv
```
#### Git Clone

To use this framework, you first need to clone the deep_active_learning repository using the following command:
```
git clone https://github.com/neuropoly/deep_active_learning.git
```
Then, go to the newly created git repository and install the requirements using the following commands:

```
pip install -r /path/to/requirements.txt
```
## Getting Started

#### Toy Datasets

#### Notebooks


## Help

If you experience issues during installation and/or use of this code, you can post a new issue on the deep_active_learning GitHub issues webpage. We will reply to you as soon as possible.

## Authors

* **Mélanie Lubrano** - [MelanieLu](https://github.com/MelanieLu)
* **Christian S. Perone** - [perone](https://github.com/perone)
* **Mathieu Boudreau** - [mathieuboudreau](https://github.com/mathieuboudreau)
* **Julien Cohen-Adad** - [jcohenadad](https://github.com/jcohenadad)

See also the list of [contributors](https://github.com/neuropoly/deep_active_learning/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License

Copyright (c) 2018 NeuroPoly, École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

We thank Shawn Mikula for sharing the histology data. Funded by the Canada Research Chair in Quantitative Magnetic Resonance Imaging [950-230815], the Canadian Institute of Health Research [CIHR FDN-143263], the Canada Foundation for Innovation [32454, 34824], the Fonds de Recherche du Québec - Santé [28826], the Fonds de Recherche du Québec - Nature et Technologies [2015-PR-182754], the Natural Sciences and Engineering Research Council of Canada [435897-2013], the Canada First Research Excellence Fund (IVADO and TransMedTech) and the Quebec BioImaging Network [5886].


