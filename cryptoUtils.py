# cryptoUtils.py
#
# Most of this code was written by Zack Jarrett
# for MATH 445: Intro to Cryptography
# at the University fo Arizona
# Spring, 2024
#
# See comments for each function to see the source
#
#
# Here's a congruence symbol for easy access: ≡

from math import floor, ceil, sqrt
import numpy as np
from sympy import factorint
from functools import reduce


class Ell_Pt:
    def __init__(self, x: int = None, y: int = None, isInf: bool = False) -> None:
        # You can only send isInfo without specifying x,y
        self.x = x
        self.y = y
        self.isInf = isInf

        if (isInf == True):
            assert (x == None and y == None)
        if (isInf == False):
            assert (x != None and y != None)

    def __str__(self) -> str:
        if self.isInf:
            return "∞"
        else:
            return f"({self.x}, {self.y})"

    # Have to do this for VSCode
    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, p: object) -> bool:
        return (
            (type(self) == type(p) and
             self.isInf == p.isInf and
             self.x == p.x and
             self.y == p.y
             )
        )

    # Pretty cheesy hash, I know. But it works.
    def __hash__(self) -> int:
        tupleRep = (self.x, self.y, self.isInf)
        return hash(tupleRep)

    def inverse(self) -> None:
        if self.isInf:
            return Ell_Pt(isInf=True)
        else:
            return Ell_Pt(x=self.x, y=-1*self.y)


def test_Ell_Pt():
    assert (Ell_Pt(x=1, y=4) == Ell_Pt(x=1, y=4))
    assert (Ell_Pt(isInf=True) == Ell_Pt(isInf=True))
    assert (Ell_Pt(x=1, y=4) != Ell_Pt(x=2, y=4))
    assert (Ell_Pt(x=1, y=4) != Ell_Pt(isInf=True))
    p = Ell_Pt(x=9, y=6)
    assert (p.inverse() == Ell_Pt(x=9, y=-6))


def test_Ell_Pt_InitFails():
    # Both of the below should fail.
    # Run them one at a time.
    Ell_Pt(x=1, y=2, isInf=True)
    Ell_Pt(isInf=False)


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


def inverse_mod(x: int, m: int):
    assert (m > 0)
    # Make sure x is positive
    x = x % m
    (d, inv_m, inv_x) = gcd(m, x)
    return (inv_x % m)

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


def test_decomposeIntoPowers():
    powerList = decomposeIntoPowers(38, 2)
    assert (powerList == [5, 2, 1])
    powerList = decomposeIntoPowers(100, 2)
    assert (powerList == [6, 5, 2])
    powerList = decomposeIntoPowers(103, 2)
    assert (powerList == [6, 5, 2, 1, 0])
    powerList = decomposeIntoPowers(101, 2)
    assert (powerList == [6, 5, 2, 0])

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


def test_GeneratorListAndInverse():
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


def test_PohligHellman():
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


# Elliptic Curves!
#
#  ell_curve_add_points takes the description of an elliptic curve
#  of the form:
#  y^2 ≡ x^3 + b*x + c mod q
#
#  So the curve is really defined by the parameters b, c, and q
#  q should be prime
def ell_curve_add_points(b: int, c: int, q: int, p1: Ell_Pt, p2: Ell_Pt):
    # Handle adding infinity:
    if (p1.isInf):
        return p2
    if (p2.isInf):
        return p1

    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y

    # Determine m
    # Recall there are two possibilities:
    if (p1 == p2):
        # Special Case for y=0, which would mean 
        # 0 in the denominator for m.
        if y1 == 0:
            return Ell_Pt(isInf=True)

        # Case where we are adding a point to itself
        numerator = (3 * (x1**2) + b) % q
        denominator = 2*y1
        inverse_denom = inverse_mod(denominator, q)
        m = numerator * inverse_denom % q
    else:
        # Case where p1 != p2
        #
        # Special case is when x1 == x2, in which case
        # the points sum to infinity
        if (x1 == x2):
            return Ell_Pt(isInf=True)

        # m = (y2-y1)/(x2-x1)
        # But recall we're in mod q so really this is:
        # (y2-y1) * ( inverse of (x2-x1) in mod q )
        delta_y = y2 - y1
        delta_x = x2 - x1
        inv_delta_x = inverse_mod(delta_x, q)
        m = (delta_y * inv_delta_x) % q

    x3 = (m**2 - x1 - x2) % q
    y3 = (m * (x1 - x3) - y1) % q

    return Ell_Pt(x=x3, y=y3)


def ell_naive_multiply_point(b: int, c: int, q: int, p: Ell_Pt, k: int):
    # Performs k*p, which is actually just:
    # p+p+p+...+p  where p is added to itself k times
    assert (k > 0)
    outPt = None
    for i in range(0, k):
        if i == 0:
            outPt = p
        else:
            outPt = ell_curve_add_points(b, c, q, outPt, p)
        print(f"{p} * {i+1} = {outPt}")

    print(f"final: {outPt}")
    return outPt

# Fast Multiply Elliptic Point


def ell_fast_multiply_point(b: int, c: int, q: int, p: Ell_Pt, k: int):
    # Does repeated point addition FAST
    assert (k > 0)
    powerList = decomposeIntoPowers(k, 2)
    greatestPower = powerList[0]
    pt_values = [p]
    for i in range(1, greatestPower+1):
        lastPt = pt_values[i-1]
        newPt = ell_curve_add_points(b, c, q, lastPt, lastPt)
        pt_values.append(newPt)

    outPt = pt_values[powerList[0]]
    for i in range(1, len(powerList)):
        addPt = pt_values[powerList[i]]
        outPt = ell_curve_add_points(b, c, q, outPt, addPt)

    #print(pt_values)
    #print(outPt)
    return outPt


def test_ell_multiply_point():
    # y^2 = x^3 + 4x + 4 mod 5
    b = 4
    c = 4
    q = 5
    p1 = Ell_Pt(x=1, y=2)
    k = 150

    p3 = ell_naive_multiply_point(b, c, q, p1, k)
    p4 = ell_fast_multiply_point(b, c, q, p1, k)

    assert (p3 == p4)


def test_ell_curve_add_points():
    # y^2 = x^3 + 4x + 4 mod 5
    b = 4
    c = 4
    q = 5
    p1 = Ell_Pt(x=1, y=2)
    p2 = Ell_Pt(x=4, y=3)
    p3 = ell_curve_add_points(b, c, q, p1, p2)
    assert (p3 == Ell_Pt(x=4, y=2))

    p1 = Ell_Pt(x=1, y=2)
    p3 = ell_curve_add_points(b, c, q, p1, p1)
    assert (p3 == Ell_Pt(x=2, y=0))



def test_BabyGiant():
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


def test_inverse_mod():
    x = 8
    m = 19
    inv_x = inverse_mod(x,m)
    assert(inv_x * x % m == 1)
    x = 12
    m = 19
    inv_x = inverse_mod(x, m)
    assert (inv_x * x % m == 1)
    x = 10
    m = 19
    inv_x = inverse_mod(x, m)
    assert (inv_x * x % m == 1)

def test_gcd():
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

    print("test_GCD Complete")


def test():
    test_inverse_mod()
    test_gcd()
    test_decomposeIntoPowers()
    test_crt()
    test_BabyGiant()
    test_GeneratorListAndInverse()
    test_PohligHellman()
    test_Ell_Pt()
    test_ell_curve_add_points()
    test_ell_multiply_point()


# Run the internal tests by running this script.
if __name__ == "__main__":
    test()
