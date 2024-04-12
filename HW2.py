import numpy as np

def gcd(a:int,b:int,start:bool = False):
    if start:
        global factors
        factors = []
    # Make sure a is bigger than b
    assert a >= b

    r = a%b
    q = int((a-r)/b)
    factors.append((q))
    if (r == 0):
        m = None
        for q in factors:
            new_m = [[0,1],[1,-q]]
            if type(m) == type(None):
                m = new_m
            else:
                m = np.dot(m,new_m)
        # gcd(a,b) = a*x+b*y
        x = m[0][0]
        y = m[1][0]
        return (b,x,y)
    else:
        return gcd(b,r)

for i in range(1,100):
    a = int("1" * i)
    b = 11
    if a > b:
        c, x, y = gcd(a, b, True)
        d = a*x+b*y
    else:
        c, x, y = gcd(b, a, True)
        d = b*x+a*y

    print(f"gcd({a},{b})={c}")
    #print(f"1={a}*({x})+{b}*({y})")
    
    #assert c == d

print("hi")
