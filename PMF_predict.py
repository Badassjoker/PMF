import numpy as np

def max_users_and_movies(filename):
  file = open(filename, 'r')

  movie_max = 0
  user_max = 0

  for line in file:
    line = line.strip()
    user_id, movie_id, rating = line.split('\t')

    try:
      user_id = int(user_id)
      movie_id = int(movie_id)
    except ValueError:
      continue

    movie_max = max(movie_max, movie_id)
    user_max = max(user_max, user_id)
  return [user_max, movie_max]


def rating_matrix(filename, max_user_number, max_movie_number):
  file = open(filename, 'r')

  R = np.zeros((max_user_number, max_movie_number)) #R is the rating matrix where rows stand for users and columns stand for moives
  I = np.zeros((max_user_number, max_movie_number)) #I is the existence matrix similar to R, but element is either 0 or 1 which means not rated or rated

  for line in file:
    line = line.strip()
    user_id, movie_id, rating = line.split('\t')

    try:
      user_id = int(user_id)
      movie_id = int(movie_id)
      rating = float(rating)
    except ValueError:
      continue

    R[user_id-1][movie_id-1] = rating
    I[user_id-1][movie_id-1] = 1.0
  return [R, I]

#################################
########### Functions ###########
#################################
def prediction(U, V, i, j, mu, bias_user, bias_movie, sum_I_user, sum_I_movie):
  if sum_I_movie[j-1] != 0:
    prediction = sum(U[i-1, :] * V[:, j-1]) - mu + bias_user[i-1]/sum_I_user[i-1] + bias_movie[j-1]/sum_I_movie[j-1]
  else:
    prediction = sum(U[i-1, :] * V[:, j-1]) + bias_user[i-1]/sum_I_user[i-1]
  return prediction


###################################
########### Main Loop #############
###################################

train_filename = 'train.txt'

# Preprocess the Rating matrix and max number of users and movies
max_user_number, max_movie_number = max_users_and_movies(train_filename)
R, I = rating_matrix(train_filename, max_user_number, max_movie_number)
mu = np.sum(R) / np.sum(I)
bias_user = np.sum(R, axis=1)
bias_movie = np.sum(R, axis=0)
sum_I_user = np.sum(I, axis=1)
sum_I_movie = np.sum(I, axis=0)

# Load the U and V matrix
U_filename = 'U_matrix.npy'
V_filename = 'V_matrix.npy'

U = np.load(U_filename)
V = np.load(V_filename)


# Load the test file
test_filename = 'test.txt'
test_file = open(test_filename, 'r')

prediction_filename = 'two_ratings_.txt'
prediction_file = open(prediction_filename, 'w')

for line in test_file:
  line = line.strip()
  user_id, movie_id, true_rating = line.split()

  try:
    user_id = int(user_id)
    movie_id = int(movie_id)
  except ValueError:
    continue

  # make the prediction
  rating = prediction(U, V, user_id, movie_id, mu, bias_user, bias_movie, sum_I_user, sum_I_movie)
  rating = round(rating, 3)
  prediction_file.write('%s\t%s\n' % (true_rating, rating))

prediction_file.close()
