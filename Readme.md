
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
    ~/document-logo-classification $ pip install -r requirements.txt

## Execution

    ~/document-logo-classification $ cd scripts
    ~/document-logo-classification/scripts $ python main_classify_document.py --doc_path path-to-dcoument --logo_dir path-to-logos-directory

The execution will return the most similar logo in the logo directory. If no similar logo is found the program will return _'unknown'_.  

**IMPORTANT:** Input document must be completely vertically alligned.
