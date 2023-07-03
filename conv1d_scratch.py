import numpy as np
import cv2
import os

# vector = np.array([1,2,3,4])
# print(vector.shape)
x = [0, 0, 3,6,8,2,1,4,7,9,0,0]
w = [4 ,0, 6, 3, 2]
w_r = np.array(w[:: -1])
s = 1
x = np.array(x)


out = []
  #iterate through the original array s cells per step
for i in range(0, int((len(x) - len(w_r))) + 1 , s):
    out.append(np.sum(x[i:i + w_r.shape[0]] * w_r))

out = np.array(out)
print(out)
out2 = []
# def conv1_(feature, kernel):
#     for i in range(len(feature))
for i in range(0, int((len(x) - len(w_r))) + 1 , s):
    out2.append(np.dot(x[i:i + w_r.shape[0]], w_r))


out2 = np.array(out2)
print("dot product method")
print(out2)