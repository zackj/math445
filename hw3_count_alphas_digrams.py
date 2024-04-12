import numpy as np
from itertools import permutations

# cipher_txt must be in rows of even numbers of characters
# followed by \n
cipher_txt = """YIFQFMZRWQFYVECFMDZPCVMRZWNMDZVEJBTXCDDUMJ
NDIFEFMDZCDMQZKCEYFCJMYRNCWJCSZREXCHZUNMXZ
NZUCDRJXYYSMRTMEYIFZWDYVZVYFZUMRZCRWNZDZJJ
XZWGCHSMRNMDHNCMFQCHZJMXJZWIEJYUCFWDJNZDIR"""

# A list of characters
alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
         'H', 'I', 'J', 'K', 'L', 'M', 'N',
         'O', 'P', 'Q', 'R', 'S', 'T', 'U',
         'V', 'W', 'X', 'Y', 'Z']

# Letter Frequencies scraped from:
# https://pi.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
frequency_tuples = [('E', 21912),
                    ('T', 16587),
                    ('A', 14810),
                    ('O', 14003),
                    ('I', 13318),
                    ('N', 12666),
                    ('S', 11450),
                    ('R', 10977),
                    ('H', 10795),
                    ('D', 7874),
                    ('L', 7253),
                    ('U', 5246),
                    ('C', 4943),
                    ('M', 4761),
                    ('F', 4200),
                    ('Y', 3853),
                    ('W', 3819),
                    ('G', 3693),
                    ('P', 3316),
                    ('B', 2715),
                    ('V', 2019),
                    ('K', 1257),
                    ('X', 315),
                    ('Q', 205),
                    ('J', 188),
                    ('Z', 128)]

digram_frequency_tuples = [('TH', 3.56),
                           ('HE', 3.07),
                           ('IN', 2.43),
                           ('ER', 2.05),
                           ('AN', 1.99),
                           ('RE', 1.85),
                           ('ON', 1.76),
                           ('AT', 1.49),
                           ('EN', 1.45),
                           ('ND', 1.35),
                           ('TI', 1.34),
                           ('ES', 1.34),
                           ('OR', 1.28),
                           ('TE', 1.20),
                           ('OF', 1.17),
                           ('ED', 1.17),
                           ('IS', 1.13),
                           ('IT', 1.12),
                           ('AL', 1.09),
                           ('AR', 1.07),
                           ('ST', 1.05),
                           ('TO', 1.05),
                           ('NT', 1.04),
                           ('NG', 0.95),
                           ('SE', 0.93),
                           ('HA', 0.93),
                           ('AS', 0.87),
                           ('OU', 0.87),
                           ('IO', 0.83),
                           ('LE', 0.83),
                           ('VE', 0.83),
                           ('CO', 0.79),
                           ('ME', 0.79),
                           ('DE', 0.76),
                           ('HI', 0.76),
                           ('RI', 0.73),
                           ('RO', 0.73),
                           ('IC', 0.70),
                           ('NE', 0.69),
                           ('EA', 0.69),
                           ('RA', 0.69),
                           ('CE', 0.65)]


count_by_alpha = {}
count_by_digram = {}

for c in alpha:
    count_by_alpha[c] = cipher_txt.count(c)

for i in range(0, len(cipher_txt)):
    if len(cipher_txt)<i+2:
        continue
    digram = cipher_txt[i:i+2]
    if '\n' in digram:
        continue

    if digram in count_by_digram.keys():
        count_by_digram[digram] += 1
    else:
        count_by_digram[digram] = 1

# We now have a count of all the letters in the cipher_text
# and a count of all the digrams in the cipher_text

# get the top 10 letters

alpha_tuples = list(((key, count_by_alpha[key]) for key in count_by_alpha.keys()))
digram_tuples = list(((key, count_by_digram[key]) for key in count_by_digram.keys()))

def sort_by_second_value(t):
    return t[1]


alpha_tuples.sort(key=sort_by_second_value, reverse=True)
digram_tuples.sort(key=sort_by_second_value, reverse=True)

# Print out the sorted tuples
# so that you can compare them with the English frequency tables
print_tables = True
if print_tables:
    print("Alpha Frequencies:")
    for t in alpha_tuples:
        print(t)

    print("")
    print("Digram Frequencies")
    for t in digram_tuples:
        print(t)

print("")

n = 26
# take the top n found characters and see what their occurence in relation
# to eachother is in the digrams
top_alphas = []
for i in range(0,n):
    c = alpha_tuples[i][0]
    top_alphas.append(c)

# This array will be organized like on page 22 of the text:
#    W  B  R   S ...
# W  3  4  12  2 
# B  4  4  0   11
# R  5  5  0   1
# S  1  0  5   0
# .
# .
# .

# Initialize values table to all zeros.
digram_table = np.zeros((n,n),dtype=np.int16)

for i in range(0,n):
    for j in range(0,n):
        # i is row index, j is column idx
        digram = alpha_tuples[i][0]+alpha_tuples[j][0]
        #print(digram)
        if digram in count_by_digram.keys():
            digram_table[i, j] = count_by_digram[digram]

if print_tables:
    print("Digram Frequency Table\n")
    h_header = "  " + " ".join(top_alphas)
    print(h_header)
    for i in range(0, n):
        this_row = top_alphas[i]
        for j in range(0, n):
            this_row = this_row + " " + str(digram_table[i,j])
        print(this_row)
    print("")


# The following is a failed attempt at brute forcing some semblance of 
# sanity on the text by translating characters with a map
# based on permutations of the top hits.
try_brute = False

if try_brute:
    transform = {}
    map_permutations = list(permutations(range(0, 6)))

    for i in range(len(map_permutations)):
        p = map_permutations[i]
        transform = {}
        for j in range(0,len(p)):
            source_char = alpha_tuples[j][0]
            map_char = frequency_tuples[p[j]][0]
            #print(source_char)
            #print(map_char)
            transform[source_char] = map_char
            
        trans_map = str.maketrans(transform)
        print(f"iteration {i}: {transform}")

        new_string = cipher_txt.translate(trans_map)
        print(new_string)
        print("-=" * 20)