import numpy as kk
Array=[[1,2,3],[4,5,6]]
B=kk.array(Array)
print(B.shape)
print(B.ndim)
print(B.size)
a=kk.array([2,4,5,7,9,11])
print(a)
print(a+5)
print(a*2);print(a-3)

Array=kk.mean(a)
print(Array)

List=[10,20,30,40,50]
Array=kk.array(List)
print(Array)