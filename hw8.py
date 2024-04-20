# hw8.py
#
# Computer Problem Solutions for Homework 8
# Computer Problems: 21.7
# Problems 1 and 4
# Written by Zack Jarrett
# April 19, 2024

from cryptoUtils import Ell_Pt, ell_add_points, ell_naive_multiply_point, ell_fast_multiply_point, ell_size_of_E_by_hasse
from sympy import primefactors
from math import floor, sqrt


def p1_A():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
    # Find the sum (1,5) + (9,3).
    print("1.a:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = Ell_Pt(x=9, y=3)
    p3 = ell_add_points(b, c, q, p1, p2)

    print(f"{p1} + {p2} = {p3}")


def p1_B():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
    # Find the sum (9,3) + (9,-3).
    print("1.b:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=9, y=3)
    p2 = Ell_Pt(x=9, y=-3)
    p3 = ell_add_points(b, c, q, p1, p2)

    print(f"{p1} + {p2} = {p3}")


def p1_C():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
    # find the difference (1, 5) - (9,3).
    print("1.c:")
    b = 2
    c = 3
    q = 19
    p1 = Ell_Pt(x=1, y=5)
    p2 = Ell_Pt(x=9, y=3)
    p3 = ell_add_points(b, c, q, p1, p2.inverse())

    print(f"{p1} - {p2} = {p3}")


def p1_D():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
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
            p2 = ell_add_points(b, c, q, p2, p1)
        # print(f"k: {k+1}, p:{p2}")
        k += 1

    # p4 = ell_naive_multiply_point(b, c, q, p1, 5)
    # q1 = p1
    # q2 = ell_curve_add_points(b, c, q, q1, p1)
    # q3 = ell_curve_add_points(b, c, q, q2, p1)
    # q4 = ell_curve_add_points(b, c, q, q3, p1)
    # q5 = ell_curve_add_points(b, c, q, q4, p1)
    # f5 = ell_fast_multiply_point(b, c, q, p1, 5)

    print(f"k={k} since {k}*{p1} == {p3}")


def p1_e():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
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
            p2 = ell_add_points(b, c, q, p2, p1)
        multiples.add(p2)
        k += 1
        print(f"{k}*{p1} == {p2}")

    print(f"{p1} has exactly k = {k} distinct multiples.")


def p1_f():
    # Let E be the elliptic curve y^2 ≡ x^3 + 2x + 3 (mod 19).
    # Using (e) and Exercise 19(d), show that the number of 
    # points on E is a multiple of 20. Use Hasse's theorem
    # to show that E has exactly 20 points.
    print("1.f:")
    b = 2
    c = 3
    q = 19
    p = Ell_Pt(x=1, y=5)
    k = 20
    l = ell_size_of_E_by_hasse(b, c, q, p, k)
    print(f"The Hasse Solver returned: {l}")


def preamble():
    print("ZACK JARRETT")
    print("MATH 445, Spring 2024, University of Arizona")
    print("Homework 8, Computer Problems 21.7.1 and 21.7.2")
    print("Code available at:")
    print("https://github.com/zackj/math445/blob/main/hw8.py")
    print("https://github.com/zackj/math445/blob/main/cryptoUtils.py\n")


def p4_A():
    # Let E be the elliptic curve y^2 ≡ x^3 - 10x + 21 (mod 557).
    # Let P = (2,3) be a point on the curve.
    # Show that 189P = ∞, but 63P ≠ ∞, and 27P ≠ ∞
    print("4.a:")
    b = -10
    c = 21
    q = 557
    p = Ell_Pt(x=2, y=3)
    k = 189
    p_189 = ell_fast_multiply_point(b, c, q, p, k)
    print(f"{k}*{p} == {p_189}")

    k = 63
    p_63 = ell_fast_multiply_point(b, c, q, p, k)
    print(f"{k}*{p} == {p_63}")

    k = 27
    p_27 = ell_fast_multiply_point(b, c, q, p, k)
    print(f"{k}*{p} == {p_27}")


def p4_B():
    # Let E be the elliptic curve y^2 ≡ x^3 - 10x + 21 (mod 557).
    # Let P = (2,3) be a point on the curve.
    # Use Exercise 20 to show P has order 189
    print("4.b:")

    b = -10
    c = 21
    q = 557
    p = Ell_Pt(x=2, y=3)
    k = 189
    k_factors = primefactors(k)
    print(f"We know from 4.a that {k} * {p} = ∞.")
    print("Exercise 20 tells us that if (k/q)*P ≠ ∞ for each prime factor q of k then k is the order of p.")
    print(f"{k} has prime factors: {k_factors}")
    for i in range(len(k_factors)):
        m = int(k/k_factors[i])
        print(f"Let m = {k}/{k_factors[i]}={m}")
        print(f"In 4.a we showed that m * p = {m} * {p} ≠ ∞")
    print(f"So we see that (k/q)*P ≠ ∞ for each prime factor of k.")
    print(f"Hence, the order of {p} is {k}")


def p4_C():
    # Let E be the elliptic curve y^2 ≡ x^3 - 10x + 21 (mod 557).
    # Let P = (2,3) be a point on the curve.
    # Use Exercise 19.d and Hasse's Theorem to show that E has 567 points.
    print("4.b:")

    b = -10
    c = 21
    q = 557
    p = Ell_Pt(x=2, y=3)
    k = 189
    l = ell_size_of_E_by_hasse(b, c, q, p, k)
    print(f"The Hasse Solver returned: {l}")

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

print("BEGIN COMPUTER PROBLEM 4\n")
p4_A()
print("")
p4_B()
print("")
p4_C()
