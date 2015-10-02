import numpy
import theano
import theano.tensor as T

rng = numpy.random
N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size = N, low = 0, high = 2))
training_steps = 1000

# Declare theano symbolic variables
x = T.matrix('x')
y = T.vector('y')
w = theano.shared(rng.randn(feats), name = 'w')
b = theano.shared(0., name = 'b')
print 'Initial model: ', w.get_value(), b.get_value()

#Construct Theano expression graph
p_1 = 1 / ( 1 + T.exp(-T.dot(x, w) - b) )               # probability that target = 1
prediction = p_1 > 0.5                                       # prediction threshold
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)    # cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()         # cost to minimize
gw, gb = T.grad(cost, [w, b])                              # compute the gradient of the cost 

# Compile 
train = theano.function(
                inputs = [x, y],
                outputs = [prediction, xent],
                updates = ((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs = [x], outputs = prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    
# Accuracy
acc = D[1] == predict(D[0])


print 'Final model: ', w.get_value(), b.get_value()
print 'target values for D: ', D[1]
print 'prediction on D: ', predict(D[0])
print 'accuracy: ', acc
