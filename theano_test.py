
import numpy
import THEANO
import THEANO.tensor as tt

rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(THEANO.config.floatX),
rng.randint(size=N,low=0, high=2).astype(THEANO.config.floatX))
training_steps = 10000

# Declare THEANO symbolic variables
x = tt.matrix("x")
y = tt.vector("y")
w = THEANO.shared(rng.randn(feats).astype(THEANO.config.floatX), name="w")
b = THEANO.shared(numpy.asarray(0., dtype=THEANO.config.floatX), name="b")
x.tag.test_value = D[0]
y.tag.test_value = D[1]

# Construct THEANO expression graph
p_1 = 1 / (1 + tt.exp(-tt.dot(x, w)-b)) # Probability of having a one
prediction = p_1 > 0.5 # The prediction that is done: 0 or 1
xent = -y*tt.log(p_1) - (1-y)*tt.log(1-p_1) # Cross-entropy
cost = xent.mean() + 0.01*(w**2).sum() # The cost to optimize
gw,gb = tt.grad(cost, [w,b])

# Compile expressions to functions
train = THEANO.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates=[(w, w-0.01*gw), (b, b-0.01*gb)],
            name = "train")
predict = THEANO.function(inputs=[x], outputs=prediction,
            name = "predict")

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
          train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if THEANO used the cpu or the gpu')
    print(train.maker.fgraph.toposort())

for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("target values for D")
print(D[1])

print("prediction on D")
print(predict(D[0]))

print("floatX=", THEANO.config.floatX)
print("device=", THEANO.config.device)