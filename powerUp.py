import numpy as np
from math import floor
from sympy import Matrix, primefactors, factorint

def decomposeIntoPowers(inVal:int, b:int):
    # inVal comes in like 38
    # we need to decompose it into the powers of base (b)
    # Suppose b is 2, this is:
    # 38 = 32 + 4 + 2 = 2^5 + 2^2 + 2^0
    # 
    # This function only works with integers greater than or equal to 0
    assert(inVal >= 0)
    assert(b >= 0)

    # make a copy of inVal to start with
    n = inVal
    powerList = list()
    
    # handle b^0 portion
    #if n%2 == 1:
    #    powerList.append(0)
    #    n = n-1
    
    # find the greatest power of b that fits in n
    i = 0
    while (b**i<=n):
        i+=1
    
    greatestPower = i-1
    powerList.append(greatestPower)
    n = n-(b**greatestPower)

    # collect all the rest of the powers of b
    for i in range(greatestPower-1,-1,-1):
        if (b**i<=n):
            powerList.append(i)
            n = n-(b**i)
    
    print(powerList)

    # n should be 0 now.
    assert(n == 0)

    # build n back up from whole cloth and make sure it matches.
    for i in powerList:
        n = n+(b**i)
    
    assert(n == inVal)

    return powerList

p = decomposeIntoPowers(10,2)

def powerUpModulo(base:int, power:int, mod:int):
    powerList = decomposeIntoPowers(power,2)
    greatestPower = powerList[0]
    modValues = [base%mod]
    for i in range(1,greatestPower+1):
        lastVal = modValues[i-1]
        newVal = (lastVal**2)%mod
        modValues.append(newVal)
    
    outVal = 1
    for i in powerList:
        outVal = (outVal*modValues[i]) % mod

    print(modValues)
    print(outVal)
    print("hi")
    return outVal

base = 123
power = 33829201
mod = 53901

base = 2718
power = 10
mod = 2776099

n = newPowerUpModulo(base, power, mod)

m = (base**power)%mod

assert (n==m)
print(n)
print(m)
print("done")