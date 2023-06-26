import numpy as np
from math import cos, sin, tan, pi

theta = 20.
c = cos(theta/180*pi)
s = sin(theta/180*pi)
mat = np.array([c, -s, 20, s, c, 10, 0, 0, 1])
mat = mat.reshape((3, 3))
invmat = np.linalg.inv(mat)
print(invmat)

p = [1, 12, 1]
p = np.array(p)
p = p.reshape((3, 1))
pp = np.matmul(invmat, p)
print(pp)