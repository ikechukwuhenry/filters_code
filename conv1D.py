import numpy as np

'''
Got alot of help from this article: https://towardsdatascience.com/convolutions-in-one-dimension-using-python-54d743f18063
By Marcello Politi
Towards Data Science
'''


def conv1D(x,filter, padding=0 , stride=1): 
  '''
  x : input vector
  filter : filter
  padding : padding size
  stride : stride
  '''
  assert len(filter) <= len(x), "x should be bigger than w"
  assert padding >= 0, "padding cannot be negative"

  filter_r = np.array(filter[::-1]) #rotation of filter 
  x_padded = np.array(x)

  if padding > 0 :
    zeros = np.zeros(shape = padding)
    x_padded = np.concatenate([zeros, x_padded, zeros]) #add zeros around original vector

  output = []
  #iterate through the original array s cells per step
  for i in range(0, int((len(x_padded) - len(filter_r))) + 1 , stride):
    output.append(np.dot(x_padded[i:i + filter_r.shape[0]], filter_r))
  
  return np.array(output)




