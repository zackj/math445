import numpy as np
from math import floor, sqrt, ceil
from scipy.linalg import null_space
from sympy import Matrix
import galois

# Problem: 2^x congruent to 9 mod 11
#          a^x congruent to b mod p
p=11
N = ceil(sqrt(p-1))
a = 2
b = 9
n_inv = 6


print(N)

babyStepList = []
for i in range(0,N):
    babyStepList.append((a**i)%p)



giantStepList = []

for i in range(0, N):
    giantStepList.append((b*(a**(n_inv*i))) % p)


print(babyStepList)

print(giantStepList)