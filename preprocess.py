import random

fp1 = open('rating.csv')
fp2 = open('train_.txt', 'w')
fp5 = open('test_.txt', 'w')

user_dict = dict()
movie_dict = dict()
user_iterator = 0
movie_iterator = 0

for line in fp1:
	line = line.strip().split(',')
	line = line[:-1]

	try:
		line[0] = int(line[0])
		line[1] = int(line[1])
	except:
		continue

	if line[0] > 3000:
		break

	if line[0] not in user_dict:
		user_iterator += 1
		user_dict[line[0]] = user_iterator

	if line[1] not in movie_dict:
		movie_iterator += 1
		movie_dict[line[1]] = movie_iterator

	user_id = user_dict[line[0]]
	movie_id = movie_dict[line[1]]

	seed = random.random()

	if seed >= 0.1:
		fp2.write('%s\t%s\t%s\n' % (user_id, movie_id, line[2]))
	else:
		fp5.write('%s\t%s\t%s\n' % (user_id, movie_id, line[2]))

fp1.close()
fp2.close()
fp5.close()

print len(user_dict), len(movie_dict)