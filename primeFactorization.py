import numpy as np
from math import floor, sqrt
from scipy.linalg import null_space
from sympy import Matrix
import galois

primes = [2, 3, 5, 7, 11, 13, 19]

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


def primeFactorization(inVal:int):
    # inVal comes in like 38
    # we need to decompose it into it's prime factors
    # 38 = 
    # 
    # This function only works with integers greater than or equal to 0
    assert(inVal >= 0)
    
    # make a copy of inVal to start with
    n = inVal

    # This array aligns with primes
    factorPowers = [0 for i in primes]
    
    for i in range(0, len(primes)):
        p = primes[i]
        if (n==1):
            break
        if n%p == 0:
            m = n
            j = 0
            while (m%p == 0):
                m = int(m/p)
                j+=1
            factorPowers[i] = j
            n = int(n/(p**j))
    
    if n!=1:
        return {"success": False, "factorPowers": None}

    m = 1
    # test that the factorization is correct
    for i in range(0, len(primes)):
        p = primes[i]
        power = factorPowers[i]
        m = m*(p**power)
    
    if (m == inVal):
        return {"success":True, "factorPowers":factorPowers}
    else:
        return {"success": False, "factorPowers": None}


# Implementation for x^2 = y^2 factorization.abs

factorResult = primeFactorization(3291028)
success = factorResult["success"]
factorPowers = factorResult["factorPowers"]

foundSquares = []

n = 3837523
for i in range(1,96):
    for j in range(1,7):
        testVal = floor(sqrt(i*n)+j)
        m = (testVal**2)%n
        factorResult = primeFactorization(m)
        success = factorResult["success"]
        factorPowers = factorResult["factorPowers"]
        
        if success:
            s = squareInfo(testVal,factorPowers,i,j,m,n)
            foundSquares.append(s)




# Put all the squares into a matrix:

#F = np.zeros(shape=(len(foundSquares), len(primes)), dtype=int)
#for i in range(0, len(foundSquares)):
#    s = foundSquares[i]
#    print(s)
#    F[i, :] = s.primePowers

# The following seeks out the known desired squares
# and puts them in the array

desiredList = [9398, 19095, 1964, 17078, 8077, 3397, 14262]
F = np.zeros(shape=(len(desiredList), len(primes)), dtype=int)
for i in range(0, len(desiredList)):
    seekBase = desiredList[i]
    for j in range(0, len(foundSquares)):
        s = foundSquares[j]
        if s.base == seekBase:
            print(s)
            F[i,:] = s.primePowers

print("F:")
print(F)

A = F%2

print("A:")
print(A)

# get the sympy matrix:
SA = Matrix(A)

# Get REF
REF = SA.echelon_form()
print("A_REF:")
print(np.array(REF))

# Get RREF
RREF = SA.rref()[0]

print("A_RREF:")
print(np.array(RREF))


NS = RREF.nullspace()

print("NS:")
print(np.array(NS))

print("done")