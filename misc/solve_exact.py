from sympy import *

x, y, z = symbols('x y z', real=True)

r2 = x + y * I
l2 = abs(r2)
r1 = l2 + z * I
r2 /= l2

for n in range(2, 11, 2):
    z1 = r1 ** n
    z2 = r2 ** n

    vx = re(z1) * re(z2)
    vy = re(z1) * im(z2)
    vz = im(z1)

    print("n = {}".format(n))

    for u, v in [(x, vx), (y, vy), (z, vz)]:
        v = together(simplify(v))
        print("{} = {}".format(u, v))
