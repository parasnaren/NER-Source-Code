# Named Entity Recogition and Category classification

## Data

The folder contains two subdirectories
- **category-data**
- **ner-data**

Each of which contains the data that were retrieved and pre-processed and then used for generating our models.

## Source

The folder contains 3 subdirectories
- **Category**
- **NER**
- **Pre-processing**

The **Category** and **NER** directories contain codes in jupyter notebooks, each implementing a different algorithm/model.
*The jupyter notebooks can be run directly accessing the data in the Data folder.*

The **Pre-processing** folder cotains the codes that were used to scrape, retrieve, parse and extract data required to build and train our models.


## Data to be downloaded

-   Download the XLNet base cased model from [here](https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip), and place it into the folder **/Source/Category/xlnet/data/**
-   Download our generated flair-embeddings model from the [drive link](https://drive.google.com/open?id=1XZWm5nGf8s_FLrJlxYamEPbPIRCYrHBH), and place it into the folder **/Source/NER/flair-ner/taggers/**
-   Download [this](https://drive.google.com/open?id=1WtilfSm9xkBwgcYNrtVBg6ojPdnYRaePs) file and place in the folder **Data/ner-data/**
-   Download [this](https://drive.google.com/open?id=1jlzvS4GF56vfxb5Wm_4-TrUgICz4KYDs) file, unzip it and place in the folder **/Data/category-data/news-files/**
-   Download [this](https://drive.google.com/open?id=1t2bs-j0ZWmh6igA9miD4d8WJaUOCF7nI) file and place in the folder **/Data/category-data/temp-files/**

