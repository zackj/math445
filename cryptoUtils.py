# cryptoUtils.py
#
# Most of this code was written by Zack Jarrett
# for MATH 445: Intro to Cryptography
# at the University fo Arizona
# Spring, 2024
#
# See comments for each function to see the source

from math import floor, ceil, sqrt
import numpy as np
from sympy import factorint
from functools import reduce

# chinese_remainder and mul_inv were
# lifted wholesale from
# https://rosettacode.org/wiki/Chinese_remainder_theorem


def chinese_remainder(n, a):
    sum = 0
    prod = reduce(lambda a, b: a*b, n)
    for n_i, a_i in zip(n, a):
        p = prod // n_i
        sum += a_i * mul_inv(p, n_i) * p
    return sum % prod


def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1:
        return 1
    while a > 1:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0:
        x1 += b0
    return x1


def test_crt():
    n = [3, 5, 7]
    a = [2, 3, 2]
    # Means:
    # x ≡ 2 mod 3
    # x ≡ 3 mod 5
    # x ≡ 2 mod 7
    x = chinese_remainder(n, a)
    # print(f"CRT solution: {x}")
    # Test the result for individual congruences
    for i in range(len(n)):
        t = x % n[i]
        assert (t == a[i])
        # print(f"{x} % {n[i]} = {a[i]}")


# GCD implementation based on:
#       The Extended Euclidean Algorithm
#       Andreas Klappenecker
#       August 25, 2006
def gcd(inA: int, inB: int):
    # Make sure a is bigger than b
    assert inA >= inB
    a = inA
    b = inB
    m = np.array([[1, 0], [0, 1], [a, b]])

    while b != 0:
        q = floor(a/b)
        q_m = np.array([[0, 1], [1, (-1*q)]])
        m = np.dot(m, q_m)
        a = m[2][0]
        b = m[2][1]

    # Return
    # gcd(a,b) = g = ar+bs
    g = m[2][0]
    r = m[0][0]
    s = m[1][0]
    if g < 0:
        g = g*(-1)
        r = r*(-1)
        s = s*(-1)
    return (g, r, s)


def decomposeIntoPowers(inVal: int, b: int):
    # inVal comes in like 38
    # we need to decompose it into the powers of base (b)
    # Suppose b is 2, this is:
    # 38 = 32 + 4 + 2 = 2^5 + 2^2 + 2^0
    #
    # This function only works with integers greater than or equal to 0
    assert (inVal >= 0)
    assert (b >= 0)

    # make a copy of inVal to start with
    n = inVal
    powerList = list()

    # handle b^0 portion
    # if n%2 == 1:
    #    powerList.append(0)
    #    n = n-1

    # find the greatest power of b that fits in n
    i = 0
    while (b**i <= n):
        i += 1

    greatestPower = i-1
    powerList.append(greatestPower)
    n = n-(b**greatestPower)

    # collect all the rest of the powers of b
    for i in range(greatestPower-1, -1, -1):
        if (b**i <= n):
            powerList.append(i)
            n = n-(b**i)

    # print(powerList)

    # n should be 0 now.
    assert (n == 0)

    # build n back up from whole cloth and make sure it matches.
    for i in powerList:
        n = n+(b**i)

    assert (n == inVal)

    return powerList

# Fast power up!


def powerUpModulo(base: int, power: int, mod: int):
    if power == 0:
        return 1

    powerList = decomposeIntoPowers(power, 2)
    greatestPower = powerList[0]
    modValues = [base % mod]
    for i in range(1, greatestPower+1):
        lastVal = modValues[i-1]
        newVal = (lastVal**2) % mod
        modValues.append(newVal)

    outVal = 1
    for i in powerList:
        outVal = (outVal*modValues[i]) % mod

    # print(modValues)
    # print(outVal)
    # print("hi")
    return outVal


# Baby Step Giant Step
# Implementation based on description in
# Introduction to Cryptography with Coding Theory, 3rd Edition
# by Wade Trappe and Lawrence C. Washington
def babyStepGiantStep(a: int, b: int, p: int):
    # The Discrete Log Problem attempts to solve for x in:
    # a^x ≡ b mod p
    #
    # in the literature:
    #   a is alpha, a generator ("primitive root" in the old language)
    #   b is beta
    #   p is prime

    N = ceil(sqrt(p-1))
    aToTheN = powerUpModulo(a, N, p)
    (g, r, s) = gcd(p, aToTheN)
    # g = p * r + aToTheN * s

    # so s is the multiplicative inverse of aToTheN

    inverseAToTheN = s

    # print(f"N: {N}; N_inv: {inverseAToTheN}")

    babyStepList = []
    for j in range(0, N):
        babyStepList.append(powerUpModulo(a, j, p))

    giantStepList = []

    success = False
    x = 0
    for k in range(0, N):
        v = (b * powerUpModulo(inverseAToTheN, k, p)) % p
        if v in babyStepList:
            j = babyStepList.index(v)
            x = j+N*k
            success = True
            # print(f"Match Found: {v}")
            break
        else:
            giantStepList.append(v)

    # print(babyStepList)
    # print(giantStepList)
    return (success, x)


def generatorListAndInverses(a: int, p: int):
    g = [(a**i) % p for i in range(1, p)]
    g_inv = []
    for s in g:
        for i in range(1, p):
            if s*i % p == 1:
                g_inv.append(i)
                break

    g_map = {}
    for i in range(0, len(g)):
        g_map[g[i]] = g_inv[i]

    return (g, g_inv, g_map)


def testGeneratorListAndInverse():
    a = 2
    p = 29
    g, g_inv, g_map = generatorListAndInverses(a, p)
    # print(g)
    # print(g_inv)
    for i in range(0, len(g)):
        assert (g[i] * g_inv[i] % p == 1)

    for i in g_map.keys():
        assert (i * g_map[i] % p == 1)


def pohligHellmanHelper(p: int, a: int, b: int, q: int, c: int):
    j = 0
    B_j = b
    n = p-1

    g, g_inv, g_map = generatorListAndInverses(a, p)

    # aToTheQ = (a**q)%p
    # (g, r, s) = gcd(p, aToTheQ)
    # so s is the multiplicative inverse of aToTheQ
    # inverseAToTheQ = s

    # z_p_minus_zero = []
    # for i in range(1,p):
    #    z_p_minus_zero.append((a**i) % p)

    coefficients = []
    while j <= c-1:
        # Calculate delta
        exp = int(n/(q**(j+1)))
        delta = (B_j**exp) % p

        # find i such that delta...
        a_j = None
        for i in range(0, q):
            exp = int(i*n/q)
            aToTheExp = (a**exp) % p
            if delta == aToTheExp:
                a_j = i
                break
        assert (a_j != None)
        coefficients.append(a_j)

        # Update B_j
        exp = a_j*(q**j)
        aToTheExp = (a**exp) % p
        a_inv = g_map[aToTheExp]

        B_j1 = (B_j*a_inv) % p
        B_j = B_j1
        j += 1

    return coefficients

# Pohlig-Hellman Algorithm
# Implementation based on description in
# Cryptography Theory and Practice
# by Stinson-Paterson


def pohligHellman(p: int, a: int, b: int):
    # The Discrete Log Problem attempts to solve for x in:
    # a^x ≡ b mod p
    #
    # in the literature:
    #   n is p-1 for p the prime in the DLP
    #   a is alpha, a generator ("primitive root" in the old language)
    #   b is beta
    #   q is a prime factor of n
    #   c is the exponent of q in the prime factorization of n
    n = p-1
    factorDict = factorint(n)

    congruenceDict = {}

    for q in factorDict.keys():
        c = factorDict[q]
        coefficients = pohligHellmanHelper(p, a, b, q, c)
        m = 0
        for i in range(0, len(coefficients)):
            x = coefficients[i]
            m = m + (x * (q**i))
        congruenceDict[q**c] = m

    # The congruence dict is congruences for some
    # unknown x
    #
    # You use the Chinese Remainder Theorm to combine
    # all the congruences to get solutions for x
    #
    # Congruence dict keys are the prime factors of n
    # IE: for p = 29
    #         n = 28 = 2^2 * 7^1
    #     so d had two keys: 4 (2^2), and 7 (7^1)
    #
    # The values for the keys are the congruence class.
    # So d[4] = 3 means:  x ≡ 3 mod 4
    # So d[7] = 4 means:  x ≡ 4 mod 7
    #
    # So by CRT: x ≡ 11
    mod_list = []
    con_list = []
    for aKey in congruenceDict.keys():
        mod_list.append(aKey)
        con_list.append(congruenceDict[aKey])

    x = chinese_remainder(mod_list, con_list)

    return x


def testPohligHellman():
    p = 29
    a = 2
    b = 18
    x = pohligHellman(p, a, b)

    # Test it:
    # a^x ≡ b
    assert ((a**x) % p == b)
    print(f"Pohlig-Hellman Solution for {a}^x ≡ {b} mod {p} is {x}")
    print(f"{a}^{x} % {p} = {b}")

    p = 41
    a = 7
    b = 12
    x = pohligHellman(p, a, b)

    # Test it:
    # a^x ≡ b
    assert ((a**x) % p == b)
    print(f"Pohlig-Hellman Solution for {a}^x ≡ {b} mod {p} is {x}")
    print(f"{a}^{x} % {p} = {b}")

def test():
    test_crt()
    testGeneratorListAndInverse()
    testPohligHellman()


def testBabyGiant():
    # Problem: 2^x congruent to 9 mod 11
    #          a^x congruent to b mod p
    a = 2
    b = 9
    p = 11
    (status, x) = babyStepGiantStep(a, b, p)
    print(f"status: {status}")
    print(f"x: {x}")
    if status:
        testVal = powerUpModulo(a, x, p)
        assert (b == testVal)


def testgcd():
    a = 15
    b = 6
    (g, r, s) = gcd(a, b)

    print(f"gcd({a},{b})={g}")
    print(f"{g}={a}*({r})+{b}*({s})")
    d = a*r+b*s
    assert g == d

    a = 2181606148950875138077
    b = 7
    (g, r, s) = gcd(a, b)

    print(f"gcd({a},{b})={g}")
    print(f"{g}={a}*({r})+{b}*({s})")
    d = a*r+b*s
    assert g == d

    print("testGCD Complete")
