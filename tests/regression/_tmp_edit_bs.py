import os
R = os.getcwd()
f = os.path.join(R, "src", "driftjax", "numerics", "banded_solve.py")
c = open(f).read()
OLD1 = [
    "    n = A.shape[0]",
    "    dr = _block_row_scales(A, B, C)  # (n,)",
    "    Ae = A * dr[:, None, None]",
    "    Be = B * dr[:-1, None, None] if n > 1 else B",
    "    Ce = C * dr[1:, None, None] if n > 1 else C",
]
NEW1 = [
    "    n = A.shape[0]",
    '    if os.environ.get("DRIFTJAX_ROW_EQUIL","0") == "1":',
    "        dr = _scalar_row_scales(A, B, C)  # (N,) per-scalar-row",
    "        dr_blk = dr.reshape(n, 3)",
    "        Ae = A * dr_blk[:, :, None]",
    "        Be = B * dr_blk[:-1, :, None] if n > 1 else B",
    "        Ce = C * dr_blk[1:, :, None] if n > 1 else C",
    "    else:",
    "        dr = _block_row_scales(A, B, C)  # (n,) legacy block-level",
    "        Ae = A * dr[:, None, None]",
    "        Be = B * dr[:-1, None, None] if n > 1 else B",
    "        Ce = C * dr[1:, None, None] if n > 1 else C",
]
OLD2 = [
    "    dr_tiled = jnp.repeat(dr, 3)",
    "    lam = (dr_tiled * y).reshape(g.shape)",
]
NEW2 = [
    '    if os.environ.get("DRIFTJAX_ROW_EQUIL","0") == "1":',
    "        lam = (dr * y).reshape(g.shape)  # dr is (N,) per-scalar",
    "    else:",
    "        dr_tiled = jnp.repeat(dr, 3)",
    "        lam = (dr_tiled * y).reshape(g.shape)",
]
print("OLD1 match:", str("\n".join(OLD1) in c))
print("OLD2 match:", str("\n".join(OLD2) in c))
c = c.replace("\n".join(OLD1), "\n".join(NEW1))
c = c.replace("\n".join(OLD2), "\n".join(NEW2))
open(f, "w").write(c)
print("edit done; gate present:", "DRIFTJAX_ROW_EQUIL" in c)
