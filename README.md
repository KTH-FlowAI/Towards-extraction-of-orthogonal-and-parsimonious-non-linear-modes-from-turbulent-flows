# Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows

## Introduction
The code in this repository features a Python implemention of $\beta$-variational autoencoders ($\beta$-VAEs) based on convolutional neural networks (CNNs) to decompose and reconstruct the urban flow fields generated through well-resolved large-eddy simulation (LES) using the open-source numerical code Nek5000. Input data are instanteous fields of streamwise velocity fluctuation component. More details about the implementation and results from the training are available in ["Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows", Hamidreza Eivazi, Soledad Le Clainche, Sergio Hoyas, Ricardo Vinuesa](https://doi.org/10.1016/j.eswa.2022.117038)(2022,*Expert Systems with Applications*)


## Pre-requisites
The code was run successfully using Tensorflow>=2.6.0, using 1 GPU for training 

## Data
The dataset used for training and testing are available in **data** folder in order to ensure the reproducibility of the results. Please, get in touch using the email address for correspondance in the paper to arrange the transfer

## Training and inference
The $\beta$-VAEs training can be performed after cloning the repository 

    git clone https://github.com/Fantasy98 Towards-extraction-of-orthogonal-and-parsimonious-non-linear-modes-from-turbulent-flows.git
    cd src
    python train_cnn_vae.py



