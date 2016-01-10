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


print("Links ---- in order to write neural networks. links is an object that holds parameters. i.e. optimization targets")
f = F.Linear(3, 2) # linear-link from 3dimentional space to 2dimensional space. mini-batchの数を指定する際によく使う.
print("Wは3x2行列T, biasが2次元. 変数は初期化済み. ランダムなので実行のたびに変化する.")
print f.W.data
print f.b.data

print("y = Wx + b. computation y.")
x = Variable(np.array([[1,2,3], [4,5,6]], dtype=np.float32))
y = f(x)
print("y: ")
print y.data


print("initialize gradients to zero ----")
f.zerograds()
y.grad = np.ones((2,2), dtype=np.float32)
y.backward()
y.backward()
print("twice backward")
print f.W.grad
f.zerograds()
y.backward()
print("initialize and backward")
print f.W.grad


print("write a model as a chain")
# chainとしてmodelを記述する
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)
# 数式を各手順にきわめて似ている
def my_forward(x):
    h = l1(x)
    return l2(h)
print("l2(l1(x))")
x = Variable(np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]], dtype=np.float32))
o = my_forward(x)
print o.data


print("write a model as a chain and reuse (into class). ----")
class MyProc(object):
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)

myProc = MyProc()
o = myProc.forward(x)
print("forward output")
print o.data
print("l1 in class")
print myProc.l1.W.data
print("l2 in class")
print myProc.l2.W.data


print("a chain for CPU/GPU migration support using Chain. ----")
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4,3),
            l2=L.Linear(3,2),
        )

    # 無名関数
    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

myChain = MyChain()

print("forward output")
o = myChain(x)
print o.data


print("a chain for CPU/GPU using ChainList which behaves like a list of links. ----")
class MyChain2(ChainList):
    def __init__(self):
        super(MyChain2, self).__init__(
            L.Linear(4, 3),
            L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self[0](x)
        return self[1](h)

myChain2 = MyChain2()
print("forward output")
o = myChain2(x)
print o.data


print("optimizer.----")
model = MyChain()
optimizer = optimizers.SGD()

# setup() prepares for the optimization given a link. tensorflowのinitilizerのようなものだろう.
optimizer.setup(model)

# 最適化には2種類の実行方法がある.1つめは以下の様に勾配をマニュアルで計算して、update()をcallする方法.
model.zerograds()
# compute gradient here.. ここで勾配をマニュアルで計算せよ.
optimizer.update() # 

# 2つめは以下の様にloss functionをupdate()の引数に渡す方法.
#def lossfun(args...):
#    return loss
#
#optimizer.update(lossfun, args...)

print("easy to use weight decay, gradient clipping. ----")
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

print("Serializer is interface to serialize or deserialize for HDF5. ----")
serializers.save_hdf5('my.model', model)
serializers.load_hdf5('my.model', model)





