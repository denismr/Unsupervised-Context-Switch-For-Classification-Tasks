# Datasets

This paper introduces four original datsets, described bellow:

 - **AedesQuinx**: binary classification task. The task is to discriminate between female _Aedes Aegypti_ and female _Aedes Quinquefasciatus_;
 - **AedesSex**: binary classification task. The task is to discriminate between female and male _Aedes Aegypti_;
 - **WBFInsects**: includes five annonymized species of flying insects, described by their wing beat frequences at varying (and annotated) temperatures;
 - **Handwritten**: features extracted from time series that represent handwritten lowercase letters _g_, _p_ and _q_, on a Wacom One tablet, from 10 authors.

We also include our pre-processed version of the [Spoken Arabic Digit](http://archive.ics.uci.edu/ml/datasets/spoken+arabic+digit) dataset.
Our pre-processing only limits the features so that all entries of the original dataset share the same number of features, and reorganizes the data as a single CSV file. We also omit which is the speaker of each entry.

If you use any of our original datasets, or our specific version of [Spoken Arabic Digit](http://archive.ics.uci.edu/ml/datasets/spoken+arabic+digit)*, please, cite our paper:

```
@article{dmr2018unsupervised,
  title={Unsupervised Context Switch for Classification Tasks on Data Streams with Recurrent Concepts},
  author={dos Reis, Denis and Maletzke, Andr{\'e} and Batista, Gustavo},
  journal={ACM/SIGAPP -- In Press},
  year={2018}
}
```

* The original authors of Spoken Arabic Digit should also be cited, if our preprocessed file was used.
