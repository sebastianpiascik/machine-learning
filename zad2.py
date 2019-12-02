from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm

import math
u = [1,2,3]
v = [-2,-1,5]

cos = dot(u,v)/norm(u)/norm(v)

print(cos)

angle = arccos(cos)
print(math.degrees(angle))