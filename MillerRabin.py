# MATH 445. Spring 2024. ZACK JARRETT
# This code implements Fast Powering Up and the Miller Rabin algorithm.
# Don't look too close, it's not pretty.

import numpy as np
from math import floor, sqrt
from scipy.linalg import null_space
from sympy import Matrix, primefactors, factorint
from cryptoUtils import powerUpModulo

class MillerRabinResult:
    def __init__(self, probPrime:bool, n: int, a: int, k: int, m: int, p_factors: dict, b_list:[int]):
        self.probPrime = probPrime
        self.n = n
        self.a = a
        self.k = k
        self.m = m
        self.p_factors = p_factors
        self.b_list = b_list

    def __str__(self):
        return f"n: {self.n}\nprobPrime: {self.probPrime}\na: {self.a}\nk: {self.k}\nm: {self.m}\np_factors: {self.p_factors}\nb_list: {self.b_list}"



def millerRabinIsPrime(n: int, a: int):
    # This function only works with n greater than 1 and odd
    assert(n > 1)
    assert(np.mod(n,2)==1)

    # Enforce 1 < a < n-1
    assert(1 < a)
    assert(a < n-1)

    p_factor_dict = factorint(n-1)
    #print("The prime factors of {} : {}".format(n-1, p_factor_dict))

    k = p_factor_dict[2]

    m = int((n-1)/(2**k))
    
    # Sanity check:
    assert (n-1 == (m * int(2**k)))

    # This array aligns with primes
    #b_0_old = np.mod(a**m,n)
    b_0 = powerUpModulo(a,m,n)
    b_list = [b_0]
    if (b_0 == 1 or b_0 == -1):
        return MillerRabinResult(True, n, a, k, m, p_factor_dict, b_list)

    for i in range(0,k):
        b_prior = b_list[-1]
        b_new = np.mod(b_prior**2,n)
        b_list.append(b_new)
        if (b_new == -1):
            return MillerRabinResult(True, n, a, k, m, p_factor_dict, b_list)
        elif (b_new == 1):
            return MillerRabinResult(False, n, a, k, m, p_factor_dict, b_list)

    return MillerRabinResult(False, n, a, k, m, p_factor_dict, b_list)
    
print("MATH 445, Spring 2024. Zack Jarrett. HW6. Problem 6. Chap 9, computer problem 13.")
print("")
n = 561
a = 2
print("test miller rabin implementation:")
result_test = millerRabinIsPrime(n, a)
print(result_test)
print("")

print("Do the real work")
print("")
n = 38200901201
a = 2
result_a2 = millerRabinIsPrime(n, a)
print(result_a2)
print("")

a = 3
result_a3 = millerRabinIsPrime(n, a)
print(result_a3)
