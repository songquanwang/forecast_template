# coding:utf-8__author__ = 'songquanwang'

import theano
import theano.tensor as T
import numpy as np

if __name__ == "__main__":
    # import math
    # x = T.dscalar('x')
    # y = x ** 2
    # gy = T.grad(y, x)
    # pp(gy)  # print out the gradient prior to optimization
    # #'((fill((x ** TensorConstant{2}), TensorConstant{1.0}) * TensorConstant{2}) * (x ** (TensorConstant{2} - TensorConstant{1})))'
    # f = theano.function([x], gy)
    # f(4)
    # # array(8.0)
    # # numpy.allclose(f(94.2), 188.4)
    # True
    #
    #
    # x = T.dscalar('x')
    # y = x ** 3
    # gy = T.grad(y, x)
    #
    # f = theano.function([x], gy)
    #
    #
    # x = T.dmatrix('x')
    # s = T.sum(1 / (1 + T.exp(-x)))
    # gs = T.grad(s, x)
    # dlogistic = theano.function([x], gs)
    # dlogistic([[0, 1], [-1, -2]])
    #
    #
    # #sequence 代表i,no_sequence  -- a1,b1
    # import theano
    # import theano.tensor as T
    # x = T.dvector('x')
    # y = x ** 2
    # J, updates = theano.scan(lambda i, a1,b1 : T.grad(a1[i], b1), sequences=np.array([1]), non_sequences=[y,x])
    # f = theano.function([x], J, updates=updates)
    # f([4, 4])
    #
    #
    # ###################
    # x = T.dvector('x')
    # y = x ** 3
    # cost = y.sum()
    # gy = T.grad(cost, x)
    # H, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
    # f = theano.function([x], H, updates=updates)
    # f([4, 4])


    ###########################
    W = T.dmatrix('W')
    V = T.dmatrix('V')
    x = T.dvector('x')
    y = T.dot(x, W)
    JV = T.Rop(y, W, V)
    f = theano.function([W, V, x], JV)
    f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0, 1])
    # array([ 2.,  2.])
    f([[2, 3], [4, 5]], [[3, 6], [2, 7]], [4, 5])
    #array([ 22.,  59.])
    print 1
    # y = x1+x2,x1+x2

    w = np.array([[2, 3], [4, 5]])
    v = np.array([[3, 6], [2, 7]])
    x = np.array([4, 5])
    y= T.dot(x, w)
    JV = T.Rop(y, w, v)
    
    
##############################
W = T.dmatrix('W')
v = T.dvector('v')
x = T.dvector('x')
y = T.dot(x, W)
VJ = T.Lop(y, W, v)
f = theano.function([v,x], VJ)
f([2, 2], [0, 1])

##################
import theano
import theano.tensor as T
import numpy as np

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print(compute_elementwise(x, w, b))

# comparison with numpy
print(np.tanh(x.dot(w) + b))


############
import theano
import theano.tensor as T
import numpy as np

# define tensor variables
X = T.vector("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

results, updates = theano.scan(lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),
          sequences=[Y, P[::-1]], outputs_info=[X])
compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=results)

# test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)

print(compute_seq(x, w, y, u, p, v))

# comparison with numpy
x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
print(x_res)

############验证
x_tm1=np.array([0,1])
y=np.array([-3,-3])
p =np.array([1,1])


x_tm1=np.array([-0.99505475, -0.99505475])
y=np.array([1,1])
p =np.array([1,1])


x_tm1=np.array([ 0.96471973,  0.96471973])
y=np.array([1,1])
p =np.array([1,1])


x_tm1=np.array([ 0.99998585,  0.99998585])
y=np.array([1,1])
p =np.array([1,1])


x_tm1=np.array([ 0.99998771,  0.99998771])
y=np.array([1,1])
p =np.array([3,3])



T.tanh(T.dot(x_tm1, w) + T.dot(y, u) + T.dot(p, v))

