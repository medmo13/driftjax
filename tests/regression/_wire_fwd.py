import os, ast
R = os.getcwd()
f = os.path.join(R, "src", "driftjax", "numerics", "banded_solve.py")
c = open(f).read()
old_f = "    import jax\n    n = A.shape[0]\n    banded, kl, ku = _blocks_to_lapack_banded(A, B, C)\n    b_flat = b.reshape(-1)"
new_f = "    import jax\n    import os\n    n = A.shape[0]\n    banded, kl, ku = _blocks_to_lapack_banded(A, B, C)\n    b_flat = b.reshape(-1)":
new_f2 = "    if os.environ.get(chr(68)+chr(82)+chr(73)+chr(70)+chr(84)+chr(74)+chr(65)+chr(88)+chr(95)+chr(82)+chr(79)+chr(87)+chr(95)+chr(69)+chr(81)+chr(85)+chr(73)+chr(76), chr(48)) == chr(49):"
print("old_f present:", old_f in c)
