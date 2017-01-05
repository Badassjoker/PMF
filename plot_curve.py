import pylab

a = []

fp = open('learning_curve.txt', 'r')

for line in fp:
	line = line.strip()
	try:
		line = float(line)
	except:
		continue

	a.append(line)

fp.close()
pylab.title('Learning Curve')
pylab.xlabel('Number of update')
pylab.ylabel('|objection function| / |movie|')
pylab.xlim([-100, 2700])
pylab.ylim([0.375, 0.75])
pylab.plot(a, color='k', linewidth=3.0)

pylab.show()