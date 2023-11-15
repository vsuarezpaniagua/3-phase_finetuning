# Combining Denoising Autoencoders with Contrastive Learning to fine-tune Transformer Models

  [![License](https://img.shields.io/static/v1?label=License&message=MIT&color=blue&?style=plastic&logo=appveyor)](https://opensource.org/license/MIT)



## Table Of Content

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [GitHub](#github)
- [Contact](#contact)
- [License](#license)




[//]: # (![GitHub repo size]&#40;https://img.shields.io/github/repo-size/AntonScheving/README-Generator?style=plastic&#41;)

[//]: # ()
[//]: # (![Top Langs]&#40;https://github-readme-stats.vercel.app/api/top-langs/?username=myusername&hide=javascript,css,scss,html&theme=tokyonight&#41;)



## Description
This repository contains the code used for the paper 
"Combining Denoising Autoencoders with Contrastive Learning to fine-tune Transformer Models". 
This work focuses on improving the fine-tuning of a Transformers model for classification tasks in NLP. 
The details can be found in the <a href="https:/..."><strong>paper</a></strong> accepted for EMNLP 2023. 

ABSTRACT

Recently, using large pre-trained Transformer models for transfer learning tasks has evolved to the point 
where they have become one of the flagship trends in the Natural Language Processing (NLP) community, 
giving rise to various outlooks such as prompt-based, adapters, or combinations with unsupervised approaches, 
among many others. In this work, we propose a 3-Phases technique to adjust a base model for a classification task. 
First, we adapt the model's signal to the data distribution by performing further training with a Denoising Autoencoder (DAE). 
Second, we adjust the representation space of the output to the corresponding classes by clustering through a Contrastive Learning (CL) method. 
In addition, we introduce a new data augmentation approach for Supervised Contrastive Learning to correct the unbalanced datasets. 
Third, we apply fine-tuning to delimit the predefined categories. 
These different phases provide relevant and complementary knowledge to the model to learn the final task. 
We supply extensive experimental results on several datasets to demonstrate these claims. 
Moreover, we include an ablation study and compare the proposed method against other ways of combining these techniques.

## Installation

To install and use this repository, follow these step-by-step instructions:

1. Clone the GitHub repository: Start by cloning the GitHub repository to your local machine using the command:

```jsx
https://github.com/vsuarezpaniagua/3-phase_finetuning.git
```

2. This package was written in Python 3.8. For the deep learning part, pytorch was used. 
To use this repository, you must verify the requirements listed in requirements.txt. 
The requirements can be installed by moving to the working directory and running the following command on the terminal 
`pip install -r requirements.txt`
3. The code has been written to be run with CUDA, automatically selected inside the code. 
It is necessary to install a version of Pythorch Compatible with CUDA and the OS. 
Follow <a href="https://pytorch.org/"><strong>this</a></strong>
link if you need to install a compatible version of Pytorch.
4. If you want to check your cuda you can do it as follows:
    1. Check cuda for windows: run the following command in the cmd "nvcc --version"
    2. Check cuda for Linux or Mac: assuming that cat is your editor run "cat /usr/local/cuda/version.txt",
    or the version.txt localization if other

## Usage
 
Here are step-by-step instructions for using this repository:

1. Move to the main folder of the repository and do the data preprocessing of the datasets. 
It can be done using the notebook `Data_preprocess.ipynb` or running the command `python Data_preprocess.py` in the terminal.
Using the notebook, you can select which datasets to be preprocessed, while the Python file does it automatically.
2. Open your terminal and navigate to the directory where the generator files are located.
3. Update the hyperparameters that you want to use in `main.py`, as well as the desired model.
4. Run the command `python main.py` to start the selected trainings. 
Each one of the calls to multiple_training in this file runs multiple trainings: 
The 3-Phases approach, the Joint version and just fine-tuning.
5. The ablations are split into multiple files, each one with a fixed combination of phases.
The hyperparameters follow the same structure and notation than in `main.py`.

## GitHub

<a href="https://github.com/vsuarezpaniagua/3-phase_finetuning"><strong>Repository</a></strong>

## Contact

Feel free to reach out to me on my email:
alejo.lopez.avila@huawei.com

## License

[![License](https://img.shields.io/static/v1?label=Licence&message=MIT&color=blue)](https://opensource.org/license/MIT)
