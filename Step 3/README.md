## Model
- Model code are all in the ``./Model/`` folder
- These include models such as..
  - [UserKNN](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/385b4f0ff1ce424c54b00a5913df87afbce5058d/Step%203/model/UserKNNCF.py)
  - [ItemKNN](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/385b4f0ff1ce424c54b00a5913df87afbce5058d/Step%203/model/ItemKNNCF.py)
  - [RP3beta](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/385b4f0ff1ce424c54b00a5913df87afbce5058d/Step%203/model/RP3beta.py)
  - [Mult-VAE](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/385b4f0ff1ce424c54b00a5913df87afbce5058d/Step%203/model/MultiVAE.py)
- Check [``Parser.py``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/385b4f0ff1ce424c54b00a5913df87afbce5058d/Step%203/model/Parser.py) for the list of hyperparameters available for each model


## Acknowledgement
- We are using the [implicit](https://github.com/benfred/implicit) library for Weighted Matrix Factorization (**WMF**)
- UserKNN, ItemKNN, RP3beta, and all their associated code are from the [RecSys 2019 Deep Learning Evaluation](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation) GitHub repository
- Mult-VAE is written in Tensorflow, by referencing the [notebook](https://github.com/dawenl/vae_cf) provided by the paper's authors

**Thank you!!** :)



## Running the Models
1. **Option 1** (Run using specific hyperparameter settings and/or your own grid search)
    - For **UserKNN**, **ItemKNN**, **RP3beta**, and **WMF**: [``train.py``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/4bd491b97ae7f6e08435fe36c830b7596e9622fb/Step%203/train.py)
      - E.g. ``python3 train.py -d "ML-100K" -m "UserKNNCF"``

    - For Mult-VAE: [``MultiVAE_train.py``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/4bd491b97ae7f6e08435fe36c830b7596e9622fb/Step%203/MultiVAE_train.py)
      - E.g. ``python3 MultiVAE_train.py -d "ML-100K" -n_epochs 200 -num_hidden 1 -beta 0.1``
      - For **Mult-VAE**, we used the following shell scripts for the experiments in our paper:
        - ML-100K: [``script_ML100K_MultiVAE.sh``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/4bd491b97ae7f6e08435fe36c830b7596e9622fb/Step%203/script_ML100K_MultiVAE.sh)
        - Amazon (Video Games): [``script_AmazonVideoGames.sh``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/9899f890eecba838aba8c99ccb94dc5d73a8bd51/Step%203/script_AmazonVideoGames.sh)

2. **Option 2** (Bayesian Hyperparameter Optimization, using [scikit-optimize](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html))
    - We used this option to automatically tune the hyperparameters for **UserKNN**, **ItemKNN**, **RP3beta**, and **WMF** in our paper
    - The code is in [``hyperOpt_train.py``](https://github.com/almightyGOSU/TheDatasetsDilemmaWIP/blob/239f550d06b3072dfe718ea658a9628ed235c787/Step%203/hyperOpt_train.py)
    - Examples
      - ML-100K: [``abc``]()
      - Amazon (Video Games): [``def``]()


## Experimental Results
- The results are stored in the ``./logs/`` folder
  - Results for a particular dataset and model can be found in ``./logs/{dataset}/{model}/``
- There are some helper files to _'sort'_ and _'gather'_ those results, i.e. [``Test``](), [``Test``](), and [``Test``]()
- For example, if we consider the **Recall @ 10** metric, there are 3 files in the ``./logs/`` folder:
  - [``___results_summary___Rec_10.txt``]() shows the model performance
  - [``___results_summary___Rec_10__bar.png``]() contains the bar plot
  - [``___results_summary___Rec_10__table.png``]() shows the relative performance of Model X (row) over Model Y (column)
    - E.g. for the Amazon (Electronics) dataset in Cluster 1, the value at (Row 1, Column 2) indicates the relative improvement of RP3beta over ItemKNN
    - Values in light green (also indicated with a *) are statistically significant with a _p-value < 0.05_
    - Values in dark green (also indicated with a **) are statistically significant with a _p-value < 0.01_


