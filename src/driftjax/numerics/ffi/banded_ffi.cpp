// DriftJax FFI banded solve: LAPACK dgbsv via XLA FFI (no Python callback).
//
// Takes prebuilt LAPACK band storage + rhs, solves in native code, returns
// the solution plus the LAPACK info code (info > 0 = singular: caller
// routes to the lstsq fallback; info < 0 never happens for valid shapes).
// The caller (Python) owns band packing, equilibration, residual checks,
// and all fallback policy; this kernel is pure numerical factorization.
//
// Buffer contract (row-major XLA layout):
//   ab_t: (N, R) float64 — band storage TRANSPOSED, i.e. band column j is
//         the contiguous row ab_t[j, :], length R = kl+ku+1, LAPACK entry
//         ab[ku+i-j, j] at offset (ku+i-j). ldab = R.
//   b:    (N,) float64 RHS, overwritten semantics handled internally
//         (solution written to a separate output buffer).
// Outputs: x (N,) float64 solution (zeros when info != 0), info (1,) int32.
//
// Fortran dgbsv_ ABI: all args by reference, 32-bit ints.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" {
void dgbsv_(int *n, int *kl, int *ku, int *nrhs, double *ab, int *ldab,
            int *ipiv, double *b, int *ldb, int *info);
}

ffi::Error BandSolveImpl(ffi::Buffer<ffi::F64> ab_t, ffi::Buffer<ffi::F64> b,
                         ffi::ResultBuffer<ffi::F64> x,
                         ffi::ResultBuffer<ffi::S32> info, int64_t kl,
                         int64_t ku) {
  auto dims_ab = ab_t.dimensions();
  if (dims_ab.size() != 2) {
    return ffi::Error::InvalidArgument("ab_t must be 2D (N, R)");
  }
  int64_t n = dims_ab[0];
  int64_t R = dims_ab[1];
  auto dims_b = b.dimensions();
  if (dims_b.size() != 1 || dims_b[0] != n) {
    return ffi::Error::InvalidArgument("b must be (N,)");
  }
  // Copy band storage into LAPACK column-major layout (R x N, Fortran order
  // == C array indexed [r + R*j]). Input rows are contiguous band columns.
  double *ab = static_cast<double *>(malloc(sizeof(double) * R * n));
  double *bx = static_cast<double *>(malloc(sizeof(double) * n));
  int *ipiv = static_cast<int *>(malloc(sizeof(int) * (n > 0 ? n : 1)));
  if (!ab || !bx || !ipiv) {
    free(ab);
    free(bx);
    free(ipiv);
    return ffi::Error::ResourceExhausted("band solve alloc failed");
  }
  const double *ab_in = ab_t.typed_data();
  for (int64_t j = 0; j < n; ++j) {
    memcpy(ab + j * R, ab_in + j * R, sizeof(double) * R);
  }
  memcpy(bx, b.typed_data(), sizeof(double) * n);
  int nn = static_cast<int>(n);
  int kkl = static_cast<int>(kl);
  int kku = static_cast<int>(ku);
  int nrhs = 1;
  int ldab = static_cast<int>(R);
  int ldb = nn;
  int lapack_info = 0;
  dgbsv_(&nn, &kkl, &kku, &nrhs, ab, &ldab, ipiv, bx, &ldb, &lapack_info);
  double *x_out = x->typed_data();
  if (lapack_info == 0) {
    memcpy(x_out, bx, sizeof(double) * n);
  } else {
    memset(x_out, 0, sizeof(double) * n);
  }
  *(info->typed_data()) = static_cast<int32_t>(lapack_info);
  free(ab);
  free(bx);
  free(ipiv);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    BandSolve, BandSolveImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()   // ab_t (N, R)
        .Arg<ffi::Buffer<ffi::F64>>()   // b (N,)
        .Ret<ffi::Buffer<ffi::F64>>()   // x (N,)
        .Ret<ffi::Buffer<ffi::S32>>()   // info (1,)
        .Attr<int64_t>("kl")
        .Attr<int64_t>("ku"));
