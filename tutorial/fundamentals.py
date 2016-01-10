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
# y は結果だけもっているわけではなく、微分の結果も保持している.backwardを呼び出すことで実行される.
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
# xによる微分
# ここでは dy/dx  = 2x -2 を x=3で計算できる. とても明快.
print x.grad

print('print retain_grad ----')
# retain grad: 中間変数の勾配を計算することもできる
# ただし、retain_grad=Trueを指定しないと、中間は削除され元の変数(ここではx)の結果しか保存されない
z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True)
# zによる微分
print("---variable ")
print x.data
print y.data
print z.data
print("---grad")
print z.grad # y = -1
print x.grad # ??? i can not understand.

print("numpy array gradient ----")
x = Variable(np.array([[1,2,3], [4,5,6]], dtype=np.float32))
y = x**2 - 2*x + 1
print x.data
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
print("dy/dx=2x-2")
print x.grad



