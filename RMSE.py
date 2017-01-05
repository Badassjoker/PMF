fp = open('two_ratings_.txt')

sq_sum = 0.0

for line in fp:
	line = line.strip().split('\t')

	try:
		line[0] = float(line[0])
		line[1] = float(line[1])
	except:
		continue

	sq_sum += pow(line[0] - line[1], 2)

rmse = pow(sq_sum/44316.0, 0.5)

print 'RMSE:', rmse