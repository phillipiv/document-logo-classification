
# document-logo-classification

## Getting Started

### Clone repository

    ~ $ git clone https://github.com/philipiv/document-logo-classification.git
    ~ $ cd document-logo-classification

### Project requirements 

It is strongly advised you work in a virtual environment.\
First step is to create one and install all necessary project requirements.
       
    ~/document-logo-classification $ virtualenv env --python=python3.6
    ~/document-logo-classification $ source env/bin/activate
    ~/document-logo-classification $ pip install -r Requirements.txt

## Execution

    ~/document-logo-classification $ cd scripts
    ~/document-logo-classification/scripts $ python main_classify_document.py --doc_path path-to-document --logo_dir path-to-logos-directory
    
Example execution:

    ~/document-logo-classification/scripts $ python main_classify_document.py --doc_path ../data/documents/abc.jpg --logo_dir ../data/logos_v2


The execution will return the most similar logo in the logo directory. If no similar logo is found the program will return _'unknown'_.  

**IMPORTANT:** Input document must be completely vertically alligned.

## Data

### Logos

A naming convention apply for logos, logo names should have the following structure:

    companyname_xyz.png
    
where _xyz_ is a three digits numbers. Each company name supports up to 1000 different logos, each of them named with a different _xyz_ number.  

A very small dataset of logos following this naming convention can be downloaded [here](https://drive.google.com/open?id=1QMOGGtwLuNJic4yHMDkREH57fPQuBwsG).

### Descriptors

To speed up computation at document classification, all logos descriptors are previously calculated and extracted to a separate file. This file is created running the script _main_create_descriptors_from_dir.py_. 

This script takes two arguments: the path to logo directory and filename for output descriptors file.  

    python main_create_descriptors_from_dir.py --logos_dir ../data/logos --descriptors ../features/logos_descriptors.joblib
    
This descriptors file can be updated with the information of additional logos using the script _main_update_descriptors.py_. 

For this, the path to an existing descriptors file must be given as an argument and the location of logos to be added; _--logo_path_ is used for individual logos and _--logo_dir_ to add all logos in a given directory. One and only one of these two options should be used.

    python main_update_descriptors.py --descriptors ../features/logos_descriptors.joblib --logo_path ../data/logos/google_004.png

A file containing the descriptors for a very small dataset of logos can be found [here](https://drive.google.com/open?id=1e0g7r08Afm4r-_b36YYUwHvVnEpIy3De).

## Disclaimer

This project was done in cooperation with the company MeetPlace GmbH.