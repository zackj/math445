# hw7_10.7.4.py
#
# Computer Problem Solutions for Homework 7
# problem 10.7.4
# Written by Zack Jarrett
# April 10, 2024

from cryptoUtils import powerUpModulo, gcd, babyStepGiantStep, pohligHellman
from sympy import factorint


def partA():
    # Show 11^(1200/q) !≡ 1 mod 1201 for q in 3,2,5.abs
    print("PART A:")
    qList = (2, 3, 5)
    a = 11
    p = 1201
    for q in qList:
        m = powerUpModulo(a, int((p-1)/q), p)
        print(f"11^(1200/{q}) ≡ {m} mod p")


def partB():
    # p = 1201
    # p-1 = 1200
    print("PART B:")
    a = 11
    p = 1201
    factorDict = factorint(p-1)
    print(f"Factors of 1200 are: {factorDict}")
    maxFactors = []
    for q in factorDict.keys():
        maxFactors.append(int((p-1)/q))

    print(f"maximal factors of {p}-1 = {p-1}: {maxFactors}")

    for q in maxFactors:
        m = powerUpModulo(a, q, p)
        print(f"11^{q} ≡ {m} mod p")

    print(f"""Notice how 11 raised to each of the maximal
    factors of (p-1) is not congruent to 1 mod 1201.""")
    print(f"""We know 1201 is prime so 11^1200 ≡ 1 mod 1201 by Fermat,
    but since 11 raised to each of the maximal factors of
    p-1 is not congruent to 1, we conclude that o(11)=p-1
    therefore 11 is a primitive root of 1201.""")


def partC():
    print("PART C:")
    a = 11
    b = 2
    p = 1201
    x = pohligHellman(p, a, b)

    # Test it:
    # a^x ≡ b
    assert ((a**x) % p == b)
    print(f"Pohlig-Hellman Solution for {a}^x ≡ {b} mod {p} is {x}")
    # print(f"{a}^{x} % {p} = {b}")


def partD():
    print("PART D:")
    # Problem: 11^x congruent to 2 mod 1201
    #          a^x congruent to b mod p
    a = 11
    b = 2
    p = 1201
    (status, x) = babyStepGiantStep(a, b, p)
    # rint(f"status: {status}")
    print(f"Using Baby Step Giant Step the solution to {
          a}^x ≡ {b} mod {p} is: x ≡ {x}")
    if status:
        testVal = powerUpModulo(a, x, p)
        assert (b == testVal)

print("ZACK JARRETT")
print("MATH 445, Spring 2024, University of Arizona")
print("Homework 7, Computer Problem 10.7.4")
print("Code available at:")
print("https://github.com/zackj/math445/blob/main/hw7_10.7.4.py")
print("https://github.com/zackj/math445/blob/main/cryptoUtils.py")
print("")

partA()
print("")

partB()
print("")

partC()
print("")

partD()
print("")
