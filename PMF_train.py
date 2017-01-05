import numpy as np
import tensorflow as tf

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

  R = np.zeros((max_user_number, max_movie_number))
  I = np.zeros((max_user_number, max_movie_number))

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


def save_matrix(filename, matrix):
  np.save(filename, matrix)


def PMF_factorization(R, I, rank, U_init, V_init, loop, fp):
  lambda_v = 0.01
  lambda_u = 0.01

  max_user_number = R.shape[0]
  max_movie_number = R.shape[1]

  graph = tf.Graph()
  with graph.as_default():
    sess = tf.Session()

    U = tf.Variable(tf.to_float(U_init))
    V = tf.Variable(tf.to_float(V_init))

    distance = tf.reduce_sum(tf.mul(tf.square(tf.matmul(U, V) - R), I))
    u_regularization = tf.reduce_sum(tf.square(U))
    v_regularization = tf.reduce_sum(tf.square(V))

    error = 0.5 * (distance + (lambda_u * u_regularization + lambda_v * v_regularization))

    if loop < 2:
      learning_rate = 0.005
    elif loop >=2 and loop < 7:
      learning_rate = 0.001
    else:
      learning_rate = 0.0005

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(error)

    number_of_rated_movies = tf.reduce_sum(I)

    RMSE = distance /  tf.to_float(number_of_rated_movies)

    init = tf.initialize_all_variables()
    sess.run(init)

    counter = 0
    for i in range(100):
      sess.run(train)
      print(sess.run(RMSE), counter)
      fp.write('%s\n' % sess.run(RMSE))

      counter += 1

    RMSE = sess.run(RMSE)
    U = sess.run(U)
    V = sess.run(V)
  return U, V, RMSE


train_filename = 'train.txt'

max_user_number, max_movie_number = max_users_and_movies(train_filename)
R, I = rating_matrix(train_filename, max_user_number, max_movie_number)
mu = np.sum(R) / np.sum(I)
bias_user = np.sum(R, axis=1)
bias_movie = np.sum(R, axis=0)
sum_I_user = np.sum(I, axis=1)
sum_I_movie = np.sum(I, axis=0)

for i in range(R.shape[0]):
  for j in range(R.shape[1]):
    if I[i][j] == 1.0:
      R[i][j] = R[i][j] + mu - bias_user[i]/sum_I_user[i] - bias_movie[j]/sum_I_movie[j]

k_rank = 10

RMSE_threshold = 0.4


U = 0.1 * np.random.normal(0, 1.0, [max_user_number, k_rank])
V = 0.1 * np.random.normal(0, 1.0, [k_rank, max_movie_number])

loop = 1

fp1 = open('learning_curve.txt', 'w')

while True:
  print 'loop: ' + str(loop)

  U, V, RMSE = PMF_factorization(R, I, k_rank, U, V, loop, fp1)

  U_filename = 'U_matrix.npy'
  V_filename = 'V_matrix.npy'

  save_matrix(U_filename,U)
  save_matrix(V_filename,V)

  loop += 1

  if RMSE < RMSE_threshold:
    break

fp1.close()