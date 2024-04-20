# hw8.py
#
# Computer Problem Solutions for Homework 8
# Computer Problems: 27.1
# Problem 1 and 4
# Written by Zack Jarrett
# April 19, 2024

from cryptoUtils import Ell_Pt, ell_curve_add_points, ell_naive_multiply_point, ell_fast_multiply_point
from sympy import factorint
from math import floor, sqrt


def p1_A():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # Find the sum (1,5) + (9,3).
    print("1.a:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = Ell_Pt(x=9, y=3)
    p3 = ell_curve_add_points(b, c, q, p1, p2)

    print(f"{p1} + {p2} = {p3}")


def p1_B():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # Find the sum (9,3) + (9,-3).
    print("1.b:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=9, y=3)
    p2 = Ell_Pt(x=9, y=-3)
    p3 = ell_curve_add_points(b, c, q, p1, p2)

    print(f"{p1} + {p2} = {p3}")


def p1_C():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # find the difference (1, 5) - (9,3).
    print("1.c:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = Ell_Pt(x=9, y=3)
    p3 = ell_curve_add_points(b, c, q, p1, p2.inverse())

    print(f"{p1} - {p2} = {p3}")


def p1_D():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # Find an integer k such that k(1,5) = (9,3).
    print("1.d:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = None
    p3 = Ell_Pt(x=9, y=3)
    k = 0
    while (p2 != p3):
        if k == 0:
            p2 = p1
        else:
            p2 = ell_curve_add_points(b, c, q, p2, p1)
        #print(f"k: {k+1}, p:{p2}")
        k += 1

    #p4 = ell_naive_multiply_point(b, c, q, p1, 5)
    #q1 = p1
    #q2 = ell_curve_add_points(b, c, q, q1, p1)
    #q3 = ell_curve_add_points(b, c, q, q2, p1)
    #q4 = ell_curve_add_points(b, c, q, q3, p1)
    #q5 = ell_curve_add_points(b, c, q, q4, p1)
    #f5 = ell_fast_multiply_point(b, c, q, p1, 5)

    print(f"k={k} since {k}*{p1} == {p3}")


def p1_e():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # Show that(1, 5) has exactly 20 distinct multiples, including oo.
    print("1.e:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = None
    multiples = set()
    k = 0
    p_inf = Ell_Pt(isInf=True)
    while (p2 != p_inf):
        if k == 0:
            p2 = p1
        else:
            p2 = ell_curve_add_points(b, c, q, p2, p1)
        multiples.add(p2)
        k+=1
        print(f"{k}*{p1} == {p2}")

    print(f"{p1} has exactly k = {k} distinct multiples.")


def p1_f():
    # Let E be the elliptic curve y^2 = x^3 + 2x + 3 (mod 19).
    # Show that(1, 5) has exactly 20 distinct multiples, including oo.
    print("1.f:")

    p = 19
    f = floor(2*sqrt(p))
    print("Hasse's Theorm tells us for n := #E we have")
    print("|n-p-1|<2*sqrt(p)")
    print(f"In our case p = {p} so floor(2*sqrt(p))={f}")
    print("We know n >= 0 since it is a magnitude.")
    print(f"So 10 <= n <= 36 by the inequality |n-{p-1}| < {f}")
    print(f"We know n >= 20 since we saw in 1.e that the order of (1,5) is k = 20")
    print(f"We know that n is a multiple of k by assumption from problem 19.d.")
    print(f"So n == 20 is the only solution. E for this curve has 20 points.")
    print(f"")
    print(f"P.S. I'm not really sure how this one counts as a 'computer' problem.")

def preamble():
    print("ZACK JARRETT")
    print("MATH 445, Spring 2024, University of Arizona")
    print("Homework 8, Computer Problems 21.7.1 and 21.7.2")
    print("Code available at:")
    print("https://github.com/zackj/math445/blob/main/hw8.py")
    print("https://github.com/zackj/math445/blob/main/cryptoUtils.py")
    print("")


preamble()

print("BEGIN COMPUTER PROBLEM 1\n")
p1_A()
print("")
p1_B()
print("")
p1_C()
print("")
p1_D()
print("")
p1_e()
print("")
p1_f()
print("")
print("END OF COMPUTER PROBLEM 1\n")
