# Cross-Domain Learning for Classifying Propaganda in Online Contents

This is the data repository for our propaganda research at the Max Planck Institute for Informatics. 
Our work investigated the cross-domain learning for propaganda detection by several methods including one proposed pair-wise ranking model.
This repository mainly consists of research data as well as the code of the 'LSTMR' model.

# Data
The data comprises the following directories and files:
All these data are organized by the 'tsv' format (fields are separated by '\t').

## News
The news dataset comes from the “Hack the News Datathon”(Please refer: https://www.datasciencesociety.net/hacknews-datathon). 
We combined and reorganized different datasets from this source, to construct an article-level news corpus (news-a.tsv) and a sentence-level corpus (news-s.tsv). 
Each of these files contains the following attributes:
1. SOURCE: all are from News.
2. LABEL: Propaganda(1) or not (0).
3. TEXT: The text of an article or a sentence.

## Speech
We collected transcripts of speeches from four politicians. 
Trump and Obama are considered as contemporary speakers, largely talking about the same or related topics. 
We consider the former as more propagandistic than the latter. 
We use Joseph Goebbels (the Nazis’ Minister of Propaganda) and Winston Churchill (the Prime Minister of the UK)as prominent figures from the World War II era, with the former being more propagandistic than the latter. 
The data is organized at two different levels of granularity: articles (speech-a.tsv) and sentences (speech-s.tsv).
1. SOURCE: The speakers.
2. LABEL: Propaganda(1) or not (0).
3. TEXT: The text of an article or a sentence.

## tweets
We combine two pre-existing collections of tweets to construct this dataset(tweets.tsv). 
We consider the Twitter IRA corpus (Edgett, 2017) as propagandistic, and the 'twitter7' data in SNAP (Yang and Leskovec, 2011) as ordinary.
1. SOURCE: SNAP or IRA.
2. LABEL: Propaganda(1) or not (0).
3. TEXT: The text of the tweets.

# Code
The code includes the proposed LSTMR model (cross-lstm-rank.py) and the corresponding data preprocessing procedure (other files in this folder).
All these codes are based on Python 3.6.
The deep learning method is based on Pytorch 1.6.
And some other packages are also necessary like sklearn, pandas, NLTK.


## Data Privacy
All data was crawled or collected from public resources in compliance with legal regulations. 
We release the data in its original form, without any masking or anonymization. 
Researchers who use the data for their own projects are responsible for observing privacy rules (e.g., when combining this data with additional data sources) and other compliance regulations.

## References
If you use this dataset, please cite the following paper:
~~~~
@inproceedings{wang2020crossdomain,
  title={Cross-Domain Learning for Classifying Propaganda in Online Contents},
  author={Wang, Liqiang and Shen, Xiaoyu and de Melo, Gerard and Weikum, Gerhard},
  booktitle={Conference for Truth and Trust Online 2020 (TTO)},
  year={2020}
}
~~~~

Copyright 2020 by Max Planck Institute for Informatics 
