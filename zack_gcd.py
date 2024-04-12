import numpy as np

factors = []

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

a = 11
b = 2
c, x, y = gcd(a, b, True)
print(f"gcd({a},{b})={c}")
print(f"1={a}*({x})+{b}*({y})")
d = a*x+b*y
assert c == d

a = 2181606148950875138077
b = 7
c, x, y = gcd(a, b, True)
print(f"gcd({a},{b})={c}")
print(f"1={a}*({x})+{b}*({y})")
d = a*x+b*y
assert c == d

print("hi")

def test_gcd():
    a = 12345
    b = 11111
    c, x, y = gcd(a, b, True)
    print(f"gcd({a},{b})={c}")

    # Check that the math works out
    d = a*x+b*y
    assert c == d


    a = 1180
    b = 482
    c, x, y = gcd(a, b, True)
    print(f"gcd({a},{b})={c}")

    d = a*x+b*y
    assert c == d


    a = 26
    b = 9
    c, x, y = gcd(a, b, True)
    print(f"gcd({a},{b})={c}")

    d = a*x+b*y
    assert c == d

# Just some scratch work:
# a = 9
# c, x, y = gcd(26, a, True)
# d = 26*x+a*y
# assert c == d
# a_inv = y
# b = 2
# new_b = (a_inv * -1 * b)%26

def find_bad_affine_ciphers():
    plainText = list(range(0, 26))

    poor_choices = []
    for a in range(1, 26):
        for b in range(0, 26):
            c, x, a_inv = gcd(26, a, True)
            new_b = (a_inv * -1 * b) % 26
            #if a == a_inv and new_b == b:
            #    print(f"a: {a}, b: {b}")
            cipher_text_1 = [(a*i+b) % 26 for i in plainText]
            cipher_text_2 = [(a_inv*i+new_b) % 26 for i in plainText]
            if cipher_text_1 == cipher_text_2:
                #print(f"a: {a}, b: {b}")
                poor_choices.append((a,b))

    print(f"Number of Poor Choices: {len(poor_choices)}")

    for (a,b) in poor_choices:
        print(f"a: {a}, b: {b}")

find_bad_affine_ciphers()