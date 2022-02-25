## Dataset Source and Preprocessing Steps
- All related information can be found here: [``Dataset Source and Preprocessing Steps.pdf``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Dataset%20Source%20and%20Preprocessing%20Steps.pdf)


## Preprocessing
- [``preprocessing.py``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/preprocessing.py)
  1. Preprocesses the dataset
  2. Derives its characteristics
  3. (Optional) Partitions the dataset into train/validation/test
- E.g. ``python3 preprocessing.py -i "ML-100K.txt" -p 1``

- You can preprocess additional datasets and/or perform your own version of preprocessing
  - You might need to modify / update the following two functions in [``util.py``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/util.py)
  - https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/util.py#L27
  - https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/util.py#L167


## Characteristics
- [``characteristics.py``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/characteristics.py)
  1. Gathers the characteristics across all datasets into a [single file](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Datasets/characteristics_all.txt)
  2. Generates two nicely formatted tables ([Table 1](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Datasets/characteristics_table_basic_detailed.txt), [Table 2](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Datasets/characteristics_table_basic_advanced.txt)) for easy viewing


## Euclidean Distance
- [``euclidean_distance.py``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/euclidean_distance.py)
  1. Derives the pairwise Euclidean distance between every pair of datasets based on their characteristics
  2. Generates a simple visualisation

![Dataset Similarities (Euclidean Distance)](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Dataset%20Similarities%20(Euclidean%20Distance).png)


## Clustering
- [``clustering.py``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/clustering.py)
  1. Performs **k-means++ clustering** using scikit-learn
     - Hyperparameters: num_clusters, iterations, random_seed
     - E.g. ``python3 clustering.py -nc 5 -iter 100 -rs 1337``
  2. Stores the clustering result
     - Version 1 (Simple): [``Clustering (5 Clusters) (Simple).txt``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Clustering%20(5%20Clusters)%20(Simple).txt)
     - Version 2 (Detailed): [``Clustering (5 Clusters) (Detailed).txt``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Clustering%20(5%20Clusters)%20(Detailed).txt)
  3. Visualises the datasets as well as the clustering result
     - [``Clustering Visualisation (5 Clusters).png``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Clustering%20Visualisation%20(5%20Clusters).png)
  4. Samples 3 datasets from each cluster (for the experiments in Step 3)
     - [``Sampled Datasets (5 Clusters).txt``](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Sampled%20Datasets%20(5%20Clusters).txt)


## Datasets & Clustering Visualisation with t-SNE (5 Clusters)
![Clustering](https://github.com/almightyGOSU/TheDatasetsDilemma/blob/5dfe4b7bc10a398861bf6a676793802662088365/Step%202/Clustering%20Visualisation%20(5%20Clusters).png)
