def gcd(a: int, b: int):
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int):
    return (a * b) // gcd(a, b)

def reduce_fraction(a: int, b: int):
    d = gcd(a, b)
    return a // d, b // d