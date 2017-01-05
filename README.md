# PMF
The dataset can be downloaded from http://files.grouplens.org/datasets/movielens/ml-20m.zip.

This PMF uses the rating file only.

The preprocess.py takes the input file which is named as rating.csv and generate two files, train.txt and test.txt.
The data in train.txt is used to train the model, while test.txt is used to test.

PMF_train.py takes train.txt and generates two files named U_matrix.npy and V_matrix.npy which contain the U and V matrix.
It will also generate a file named learning_curve.txt which contains the changes of the fit error of the model.

PMF_predict.py will take the two matrix files and test.txt file to predict the ratings according to test.txt.
It will also generate a file named two_ratings.txt which contain the ground truth and the predicted ratings.

At last, RMSE.py can calculate the RMSE of the predicted ratings according to the ground truth.

Optionally, plot_curve.py can plot the learning curve using learning_curve.txt.

The library tensorflow is a must to run the program.
