# Career-Longevity-prediction-for-NBA-Rookie
Import Libr
Imported necessary libraries
## Introduction

The National Basketball Association (NBA) is a professional basketball league in North America. The league is composed of 30 teams (29 in the United States and 1 in Canada) and is one of the four major professional sports leagues in the United States and Canada. It is the premier men's professional basketball league in the world.
Career longevity is dependent on various factors for any players in all the games and so for NBA Rookies. The factors like games played, count of games played, and other statistics of the player during the game.
## Objective
Using machine learning techniques determine if a player’s career will flourish or not
## Data
The dataset contains player statistics for NRB Rookies. There are 1100+ observations in the train dataset with 19 variables excluding the target variable (i.e. Target).

GP: Games Played (here you might find some values in decimal, consider them to be the floor integer, for example, if the value is 12.789, the number of games played by the player is 12

The values for given attributes are averaged over all the games played by players

MIN:  Minutes Played

PTS: Number of points per game

FGM: Field goals made

FGA: Field goals attempt

FG%: field goals percent

3P Made: 3 point made

3PA: 3 points attempt

3P%: 3 point percent

FTM: Free throw made

FTA: Free throw attempts

FT%: Free throw percent

OREB: Offensive rebounds

DREB: Defensive rebounds

REB: Rebounds

AST: Assists

STL: Steals

BLK: Blocks

TOV: Turnovers

Target: 0 if career years played < 5, 1 if career years played >= 5
## Data Scaling : RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance like StandardScaler.

Outliers can often influence the sample mean and variance. RobustScaler which uses the median and the interquartile range often gives better results as it gave for this dataset
## Model

Create Baseline Machine Learning Model for Binary Classification Problem. Here we have reached Modelling. The most Interesting and Exciting part of the whole Hackathon to me is Modelling but we need to understand it is only 5-10 % of the Data Science Lifecycle.
## Machine Learning Models
1. LightGBM and its Hyperparameters
1. What is LightGBM?

LightGBM is a gradient boosting framework that uses tree based learning algorithm.
2. How does it differ from other tree-based algorithms? 

LightGBM grows trees vertically while other algorithms grow trees horizontally meaning that this algorithm  grows tree leaf-wise (row by row) while other algorithms grow level-wise.
3. How does it Work?

It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss (in that it chooses the leaf it believes will yield the largest decrease in loss) than a level-wise algorithm but is prone to over-fitting.
LightGBM is faster than XGBoost and it is 20 times faster with the same performance is what LightGBM’s creators claim.

Key LightGBM Hyperparameter(s) Tuned in this Hackathon:
1. scale_pos_weight=2.5
scale_pos_weight, default = 1.0, type = double, constraints: scale_pos_weight > 0.0

used only in binary and multiclassova applications
weight of labels with positive class
Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them
Only 8.5% (4668 out of total 54,808) of Employees were recommended for promotion based on Train data. 

scale_pos_weight = 54,808 / 4668 but since we have a huge imbalance we need to take the Square root of √ (54,808 / 4668) = 3.42. We can start from 3.42 to 1 unit below and above so we can cover a range of values from 2.42 to 4.42. We can finalize with 2.5 as it gave good results.
2. boosting_type = ‘dart’
boosting_type default = gbdt, type = enum, options: gbdt, rf, dart, goss, aliases: boosting_type, boost

gbdt, traditional Gradient Boosting Decision Tree, aliases: gbrt (  Stable and Reliable )
rf, Random Forest, aliases: random_forest
dart, Dropouts meet Multiple Additive Regression Trees ( Used ‘dart’ for Better Accuracy as suggested in Parameter Tuning Guide for LGBM for this Hackathon and worked so well though ‘dart’ is slower than default ‘gbdt’ )
goss, Gradient-based One-Side Sampling
Note: internally, LightGBM uses gbdt mode for the first 1 / learning_rate iterations
3. n_estimators=494 
As per the Parameter Tuning Guide for LGBM for Better Accuracyused small learning_rate with large num_iterations.

num_iterations , default = 100, type = int, aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators, constraints: num_iterations >= 0

number of boosting iterations
Note: internally, LightGBM constructs num_class * num_iterations trees for multi-class classification problems
4. learning_rate=0.15
learning_rate , default = 0.1, type = double, aliases: shrinkage_rate, eta, constraints: learning_rate > 0.0

shrinkage rate
in dart, it also affects on normalization weights of dropped trees.
5. max_depth=5
max_depth , default = -1, type = int

To deal with over-fitting restrict the max depth of the tree model when data is small. The tree still grows leaf-wise

< = 0 means no restriction

 

2. XGBoost and its Hyperparameters
1. What is XGBoost?

XGBoost (eXtreme Gradient Boosting)  is an implementation of gradient boosted decision trees designed for speed and performance.
XGBoost is an algorithm that has recently been dominating machine learning Kaggle competitions for tabular data.
2. How it differs from other tree-based algorithms?

XGBoost makes use of a greedy algorithm (in conjunction with many other features).
3. How does it Work?

XGboost has an implementation that can produce high-performing model trained on large amounts of data in a very short amount of time.
XGBoost wins you Hackathons most of the times, is what Kaggle and Analytics Vidhya Hackathon Winners claim!

Key XGBoost Hyperparameter(s) Tuned in this Hackathon
1. subsample = 0.70
subsample default=1

Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
range: (0,1}
2. updater =”grow_histmaker”
updater default= grow_colmaker,prune

A comma-separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updaters exist:
grow_colmaker: non-distributed column-based construction of trees.
grow_histmaker: distributed tree construction with row-based data splitting based on the global proposal of histogram counting.
grow_local_histmaker: based on local histogram counting.
grow_quantile_histmaker: Grow tree using a quantized histogram.
grow_gpu_hist: Grow tree with GPU.
sync: synchronizes trees in all distributed nodes.
refresh: refreshes tree’s statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
prune: prunes the splits
3. base_score=0.2
base_score default=0.5

The initial prediction score of all instances, global bias
For a sufficient number of iterations, changing this value will not have too much effect.
 

3. CatBoost and its Hyperparameters :
1. What is CatBoost?

CatBoost is a high-performance open source library for gradient boosting on decision trees.
CatBoost is derived from two words Category and Boosting.
2. Advantages of CatBoost over the other 2 Models?

Very High performance with little parameter tuning as you can see the above code compared to other 2
Handling of Categorical variables automatically with a Special Hyperparameter “cat_features“.
Fast and scalable GPU version with CatBoost.
In my experiments with Hackathons and Real world data, Catboost is the Most Robust Algorithm among the 3, check the score below for this Hackathon too.
3. How does it work better?

CatBoost can handle categorical variables through 6 different methods of quantization, a statistical method that finds the best mapping of classes to numerical quantities for the model.
CatBoost algorithm is built in such a way very less tuning is necessary, this leads to less overfitting and better generalization overall.
Key CatBoost Hyperparameter(s) Tuned in this Hackathon :
1. subsample = 0.085
Also known as “sample rate for bagging” can be used if one of the following bootstrap types is selected :
Poisson
Bernoulli
MVS
The default value depends on the dataset size and the bootstrap type:
Datasets with less than 100 objects, default = 1
Datasets with 100 objects or more and :
Poisson, Bernoulli —  default = 0.66
MVS — default = 0.80
By default, the method for sampling the weights of objects is set to “Bayesian”. The training is performed faster if the “Bernoulli” method is set and the value for the sample rate for bagging is smaller than 1
## Ensemble with Voting Classifier to Improve the  – “F1-Score” and Predict Target “is_promoted”
Max Voting using Voting Classifier: Max voting method is generally used for classification problems. In this technique, multiple models are used to make predictions for each data point. The predictions by each model are considered as a ‘vote’. The predictions which we get from the majority of the models are used as the final prediction. 


Example: If we ask 5 of our Readers to rate this Article (out of 5): We’ll assume three of them rated it as 5 while two of them gave it a 4. Since the majority gave a rating of 5, the final rating of this article will be taken as 5 out of 5. You can consider this similar to taking the mode of all the predictions.

Soft Voting : In soft voting, the output class is the prediction based on the average of probability given to that class.
Improved F1-score to 0.7 
