# Problem 2.12: Find all 28 affine ciphers for which
# the decryption algorithm is the same as the encryption
# algorithm. There are 28 of them.

def multiplicativeInverseMod26(a):
    pass

plainText = list(range(0, 26))

for a in range(0,26):
    for b in range(0,26):
        cipherText = [(a*i+b)%26 for i in plainText]
        if cipherText == plainText:
            print(f"a: {a}, b: {b}")