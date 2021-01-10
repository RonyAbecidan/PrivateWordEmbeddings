## Study of Private Word Embeddings

This repository is made as part of an assignment for the "Privacy Preserving Machine Learning" class of the [University of Lille's Msc. in Data Science](http://bit.ly/MasterDSULille).

#### Authors : 
- Samy Zouhri
- Rony Abecidan

***

Here, we are studying the paper [**"Differentially Private Representation for NLP : Formal Guarantee and An Empirical Study on Privacy and Fairness"**](https://arxiv.org/abs/2010.01285) written by Lingjuan Lyu, Xuanli He and Yitong Li and published for the 2020 Conference on Empirical Methods in Natural Language Processing. 

In this article, the authors propose for the first time a method enabling to guarantee formally the privacy of a word embedding while maintaining a satisfying utility and wiping off discriminations in most cases.

***

This repo is made of **3** parts :

- The article studied in a .pdf format

- A short report discussing about the strategy proposed in the paper for making a private word representation while maintaining utility in NLP models. There are also some additional information enable to better understand the logic of the paper.

- Two illustrative notebooks in which we propose two experiments :
     - The first one consists in studying to what extent a word embedding can leak sensitive information. We have used for this experiment [the Word2Vec embedding of the library gensim](https://radimrehurek.com/gensim/models/word2vec.html)
     - The second one consists in implementing the strategy proposed by the authors while studying its impact on utility for different classifications tasks. This time we have considered a "custom" embedding using pytorch.

***

## Installation

If you want to reproduce our experiments you'll have to install the requirements listed in requirements.txt. 

```bash
pip install -r requirements.txt
```

A part of the code for the second experiment is inspired by [the tutorial of pytorch for text classification](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
