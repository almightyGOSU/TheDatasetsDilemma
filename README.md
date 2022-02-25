### Code for the WSDM 2022 paper
### The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?

<br>

Hello! :)

This repository contains the source code, as well as other useful information, for the paper "__The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?__" in WSDM 2022.

The paper is available here: [Paper](https://dl.acm.org/doi/abs/10.1145/3488560.3498519) (**Best Paper Award Runner-up**)

For a quick overview of the paper, you can refer to these slides:
[The Datasets Dilemma Slides](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/d0e49eb91b11a522835f0fe6e24396b24fd509b9/WSDM%202022%20-%20The%20Datasets%20Dilemma%20-%20Slides.pdf)


## Reference

Please consider citing our work if you find it useful, thank you!

```
@inproceedings{10.1145/3488560.3498519,
  author = {Chin, Jin Yao and Chen, Yile and Cong, Gao},
  title = {The Datasets Dilemma: How Much Do We Really Know About Recommendation Datasets?},
  year = {2022},
  isbn = {9781450391320},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3488560.3498519},
  doi = {10.1145/3488560.3498519},
  booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
  pages = {141–149},
  numpages = {9},
  keywords = {datasets, item recommendation, evaluation, data characteristics},
  location = {Virtual Event, AZ, USA},
  series = {WSDM '22}
}
```


## Outline
In our paper, we try to address the "datasets dilemma" using 3 main steps.
1. ***How*** are different datasets being utilised in recent papers?
   - Are there any patterns?
   - Code can be found in the ``./Step 1/`` folder (Please refer to its [README file](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/e8a90f61977db22e879f2b28f2c647df329e77ae/Step%201/README.md))
2. ***What*** are the similarities as well as differences between various datasets?
   - Can we define them using objective measures?
   - Code can be found in the ``./Step 2/`` folder (Please refer to its [README file]())
3. ***If*** the choice of datasets used could influence the observations and/or conclusions obtained
   - Empirical study using a variety of item recommendation algorithms
   - Code can be found in the ``./Step 3/`` folder (Please refer to its [README file]())

**The ``./Datasets/`` folder**
- ``./Datasets/Source/`` contains the raw datasets
- ``./Datasets/Preprocessed/`` contains the preprocessed datasets
- The dataset characteristics (as well as other information) for all 51 datasets: [characteristics_all.txt](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/194e3d888a753d68ffd0c05d777d68a339a2ca6b/Datasets/characteristics_all.txt)
  - Basic Dataset Information (in a table format): [characteristics_table_basic_detailed.txt](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/194e3d888a753d68ffd0c05d777d68a339a2ca6b/Datasets/characteristics_table_basic_detailed.txt)
  - Dataset Characteristics (in a table format): [characteristics_table_basic_advanced.txt](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/194e3d888a753d68ffd0c05d777d68a339a2ca6b/Datasets/characteristics_table_basic_advanced.txt)


## Environment Setup

1. Python 3.6.8
2. PyTorch 1.4.0
3. Tensorflow 2.3.0
4. numpy 1.17.2
5. pandas 0.25.3
6. matplotlib 3.3.2
7. scikit-learn 0.23.2
8. scipy 1.3.0
9. scikit-optimize 0.8.1
10. mlxtend 0.18.0 (for frequent itemset mining)
11. implicit 0.4.4 (for Weighted Matrix Factorization (**WMF**))

Analyses & experiments were conducted on a Ubuntu server with version 16.04.6 LTS, and conda 4.8.4.
