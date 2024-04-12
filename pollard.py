import numpy as np
from math import floor, sqrt
from scipy.linalg import null_space
from sympy import Matrix


class squareInfo:
    def __init__(self, base:int, primePowers:[int], i:int, j:int, val:int, modulus:int):
        self.base = base
        self.primePowers = primePowers
        self.i = i
        self.j = j
        self.val = val
        self.modulus = modulus
    
    def __str__(self):
        return f"{self.base} | {self.i} {self.j} | {self.primePowers} | {self.val}"


def pollard(a:int, bound:int, n:int):
    b_vals = []
    powers = []
    for i in range(1,bound+1):
        last_b = a if i==1 else b_vals[-1]
        x = last_b**i
        b = x%n
        b_vals.append(b)
        powers.append(x)
        if (b==1):
            print(f"Encountered 1: {i}")
            break
    
    last_b = b_vals[-2]
    d = np.gcd(last_b, n)
    
    ret = {"success": None, "gcd": d, "b_vals": b_vals}
    if (d == n or d == 1):
        ret["success"] = False
    else:
        ret["success"] = True
    
    return ret

a = 2
bound = 20
n = 35118
n = 55

p_result = pollard(a,bound,n)

print(p_result)