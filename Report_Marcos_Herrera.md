# Shot Type Classification for Ads
## Introduction
    In  this final project a computer vision model had to be trained, in this case we had create all the python scripts to make the model work and run it inside a docker container. All the files were pushed to a GitHub repository and pulled from the server to start training. Seven experiments with different settings were run in the server, the notebook 'Model Evaluation' was used to test the training model using the weights from the first four experiments, finally these results were compared to a state of the art solution.

## Hardware AWS cloud server
    Processor: -
    GPU : Nvidia Tesla - K80 RAM: 12.0 GB 
    OS: Ubuntu


## Trainings
Bellow are shown the model parameter of each experiment.

| Exp.       | Baseline  | Description                                                | CS    | ECS   | FS    | LS    | MS    | Motion | Pull | Push | Static | Average |
|------------|-----------|------------------------------------------------------------|-------|-------|-------|-------|-------|--------|------|------|--------|---------|
| 0          | Imagenet  | GRU, train dense                                           | 77.11 | 87.56 | 85.32 | 91.86 | 82.48 | 56.58  | 0    | 0    | 74.26  | 61.69   |
| 1          | Imagenet  | LSTM, train dense                                          | 79.84 | 88.19 | 82.97 | 93.69 | 84.57 | 59.64  | 0    | 0    | 73.85  | 62.53   |
| 2          | Imagenet  | Idem 001, no regularization                                | 81.09 | 86.5  | 87.87 | 88.84 | 82.3  | 57.66  | 0    | 0    | 74.57  | 62.09   |
| 3          | Imagenet  | Idem 001, data augmentation & dropout                      | 83.2  | 86    | 83.05 | 93.04 | 81.41 | 60.23  | 0    | 0    | 73.38  | 62.26   |
| 4          | Model 003 | Finetuning                                                 | 85.23 | 83.42 | 87.57 | 91.12 | 83.11 | 48.87  | 0    | 0    | 76.74  | 61.78   |
| 5          | Model 003 | Finetuning, oversampling                                   | 82.94 | 79.81 | 82.38 | 88.68 | 78.37 | 56.74  | 0    | 3.48 | 70.78  | 60.35   |
| 6          | Model 003 | Finetuning, over & under sampling                          | 85.3  | 65.83 | 86.58 | 85.24 | 78.99 | 44.76  | 1.6  | 2.33 | 70.05  | 57.85   |
| 7          | Model 003 | Finetuning, enhanced over/under sampling, synthetic videos | 86.08 | 76.46 | 78.03 | 88.12 | 81.4  | 43.93  | 0    | 4.04 | 79.48  | 59.73   |
| Ref.  | -  |  Benchmark(state of the art)                                           | 84.5  | 85.8  | 87    | 95.1  | 86.9  | 78.5   | 27.4 | 28.4 | 95     | 74.29   |
    
    Experiment_0: The CNN Resnet50 model was loaded with the 'imagenet' weights (transfer learning), a GRU for the recursive layer and only trained dense layers, and our custom dataset (movie-shot-trailers) returning a model ready for performing finetuning. 
    The model had an acceptable learning capability as the model's train metrics were sufficiently good. Nevertheless, it did not imporvement in any manner on vlaidation data, failing to generalize.The dataset imbalance was clear on the movenet "Push" and "Pull" classes which had no predictions.

    Experiment_1: This experiment is a replica of experiment_0, the only difference is that LSTM (Long short-term memory) was used instead of GRU, the results were slightly better than with a GRU RNN.

    Experiment_2: This experiment has the same model parameters as experiment_1, except it had no regularization on the movement dense layer, despite droping the regularization, the model was not able to to improve its performance on movement "Push" and "Pull" classes even on train.

    Experiment_3: This experiment has the same parameters as experiment_1 plus data augmentation and activated dropout layers. This experiment succeded in making the validation metrics show a converging behaviour, the accuracy values achieved on the scale prediction are already very close to the ones achieved in the benchmark paper. The dataset imbalance was again clear on the movenet "Push" and "Pull" classes which had no predictions.

    Experiment_4: In this experiment a finetuning of the convolutional layer was performed using the best model achieved up to this point, which in this case is experiment 3, the results were almost the same than the ones achieved without the finetuning (experiment 003).

    Experiment_5: Finetuning of the convolutional layer using the best model achieved up to this point (experiment 3) with oversampling of minority movement classes, both scale and movement metrics slightly decreased compared to experimet 003. Oversampling failed to improve the movement F1 score, mainly related to the "Push" & "Pull" imbalanced classes.

    Experiment_6: Finetuning of the convolutional layer using the best model achieved up to this point (experiment 3) with overs and under sampling, movement metrics also had F1 score setback. Given that now the model is receiving a balanced dataset, it is exposed to less samples of the previous majority classes, which now have lower accuracies. On top of that the model was still unable to learn enough on the prior minority classes.

    Experiment_7: Finetuning of the convolutional layer using the best model achieved up to this point (experiment 3) with Push & Pull synthetic videos and Over/under samplig, for this experiment, the code structure was enhanced so that the undersampling would take place within the iterable (data generator) that provides the data to the model on each epoch, instead of doing so only once on the whole databaseand discarding the rest of the samples for the whole training.
    Basically, now the data generator shuffles and resamples the majority classes on each epoch, showing different subsets of the majority classes to the model on each epoch, the previous strategy brought the majority classes metrics back up, to almost the same ones achieved by baseline model form experient 003. However, sythec vieos failed to improve the metrics for minority movement classes.

## Improvements

1. Neural network improvements:
    1. Split lower layers not just dense ones for each prediction
    2. Subject guidance. Split subject and background for scale an movement
2. Improve data management
    1. Enable Tensorflow’s cache & prefetch to imporve training times and hyperparameters avaialbility
    2. Save Images, pickles, csv (can improve training times given the use of a custom data generator
    3. Model to select important frames from the vido instead of picking equaly separeted ones.
    4. Manually add scale labels to V2 and synthetic videos
    5. Data augmentation fine-tuning
    6. Under sampling: iterable subsets (equivalent to cross validation) instead of random sampling on each epoch
    7. Improve Video synthesizer realism, uncentered zoom, dynamic zoom video wise


## Conclusion

The fact that the model was unable to significantly improve ts moment minority clsses predictions despite under/oversampling with synthetc videos points suggests that either the model structure is not good a good aide afor the problem or that the synthetic videos are not representative for real ones.

Given how minority classes metrics remaind almos unchanged after using the synthetic videos, it seems the earlier would be more likely. Given the benchmark approach, future experiments would aim to split each prediction pipeline progresively and sophisticate the feature extraction layer:
a. Split Recursive layer, one layer for each prediction
b. Split Convolutional layer, one layer for each prediction
c. Add subject detection layer so that the subject is used for scale prediction and background for movement prediction.
