# encoding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

y = x**2 - 2 *x + 1

print('print y data, backward() 1st ----')
print type(y)
print y.data
print y.backward()

x_data = np.array([3], dtype=np.float32)
x = Variable(x_data)

# これはyが再定義されたことになる
y = x**2 - 2*x + 1

print('print y data, backward() 2nd ----')
print type(y)
print y.data
print y.backward()


print('print x grad ----')
# 勾配とその保存先は変数のgrad属性(gradient is computed and stored in the grad attribute of the input variable)
print x.grad


