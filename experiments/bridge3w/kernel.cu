// The package's merge rounds with the bridge found by a walk from the shared point for at
// most T_r = 8 (r + log2 C) steps on round r, then the three-case search over edges (Gaede,
// Goertz, van der Hoog, Krogh, Rotenberg, ALENEX 2024, mirrored to lower hulls) in
// O(log |L| + log |R|) steps, bisecting first over segments, then within one. The scheduling
// is the package's: a thread per merge in rounds of many merges, where the three-case search
// is used, and a warp per merge after that, with the package's own warp-parallel search.
#include "tiles.cuh"

// A named namespace: see experiments/README.md.
namespace tiexp_bridge3w {

using namespace torchisotonic;

struct Segs {  // a node's segment list
  const Seg* p;
  __device__ __forceinline__ Seg operator[](int i) const { return p[i]; }
};

// One side of a merge: its segment list, restricted to [s_lo, s_hi] (segments) and, once
// one segment is left, positions [p_lo, p_hi]. The shared point is excluded through a cap:
// in segment cap_seg positions are at most cap_pos (L's last segment) or at least floor_pos
// (R's first); cap_seg = -1 when the segment holding the shared point is left out entirely.
struct Side {
  Segs seg;
  int s_lo, s_hi;  // segment range, inclusive
  int p_lo, p_hi;  // position range within segment s_lo == s_hi, inclusive
  int cap_seg, cap_pos;      // positions <= cap_pos in cap_seg (L)
  int floor_seg, floor_pos;  // positions >= floor_pos in floor_seg (R)

  __device__ int hi_of(int k) const { return k == cap_seg ? cap_pos : seg[k].hi; }
  __device__ int lo_of(int k) const { return k == floor_seg ? floor_pos : seg[k].lo; }
  __device__ bool single_segment() const { return s_lo == s_hi; }
  __device__ bool fixed() const { return s_lo == s_hi && p_lo == p_hi; }
  __device__ void settle() {  // one segment left: its position range
    p_lo = lo_of(s_lo);
    p_hi = hi_of(s_hi);
  }
  // The probe edge: its two vertices (segment, position).
  __device__ void probe(int* sa, int* pa, int* sb, int* pb) const {
    if (!single_segment()) {  // the edge from the middle segment's last vertex to the next's first
      const int k = (s_lo + s_hi) / 2;
      *sa = k;
      *pa = hi_of(k);
      *sb = k + 1;
      *pb = lo_of(k + 1);
    } else {
      const int k = (p_lo + p_hi) / 2;
      *sa = *sb = s_lo;
      *pa = k;
      *pb = k + 1;
    }
  }
  __device__ void keep_before(int sa, int pa) {  // the answer is at or before vertex (sa, pa)
    if (!single_segment()) {
      s_hi = sa;
      if (single_segment()) {
        settle();
        p_hi = pa;
      }
    } else {
      p_hi = pa;
    }
  }
  __device__ void keep_after(int sb, int pb) {  // the answer is at or after vertex (sb, pb)
    if (!single_segment()) {
      s_lo = sb;
      if (single_segment()) {
        settle();
        p_lo = pb;
      }
    } else {
      p_lo = pb;
    }
  }
  __device__ void first(int* s, int* p) const {  // when fixed: the vertex
    *s = s_lo;
    *p = p_lo;
  }
};

// sign(slope(pq) - slope(uv)), x(p) < x(q), x(u) < x(v).
__device__ __forceinline__ int slope_cmp(const Pt& p, const Pt& q, const Pt& u, const Pt& v) {
  const double lhs = (q.y - p.y) * (v.x - u.x), rhs = (v.y - u.y) * (q.x - p.x);
  return (lhs > rhs) - (lhs < rhs);
}

// sign(x(gamma) - m), gamma the intersection of lines a1a2 and b1b2 (slopes differ).
__device__ __forceinline__ int gamma_x_cmp(
    const Pt& a1, const Pt& a2, const Pt& b1, const Pt& b2, double m) {
  const double dx1 = a2.x - a1.x, dy1 = a2.y - a1.y, dx2 = b2.x - b1.x, dy2 = b2.y - b1.y;
  const double den = dx1 * dy2 - dy1 * dx2;
  const double t = (b1.x - a1.x) * dy2 - (b1.y - a1.y) * dx2;
  const double v = (a1.x - m) * den + t * dx1;
  const int s = (v > 0) - (v < 0);
  return den > 0 ? s : -s;
}

// The bridge between L (segments [0, nl), the shared point its last vertex) and R (the shared
// point its first), by the three cases; (sa, pa) in L, (sb, pb) in R.
template <bool W>
__device__ void bridge_three_cases(
    const Row<W>& row, Segs L, int nl, Segs R, int nr, int* out_sa, int* out_pa, int* out_sb,
    int* out_pb) {
  // The shared point is L's last vertex and R's first: excluded by a cap on each side.
  Side A{L, 0, nl - 1, 0, 0, nl - 1, L[nl - 1].hi - 1, -1, 0};
  Side B{R, 0, nr - 1, 0, 0, -1, 0, 0, R[0].lo + 1};
  if (L[nl - 1].lo == L[nl - 1].hi) {  // the last segment is the shared point alone
    A.s_hi = nl - 2;
    A.cap_seg = -1;
  }
  if (R[0].lo == R[0].hi) {
    B.s_lo = 1;
    B.floor_seg = -1;
  }
  if (A.single_segment()) {
    A.settle();
  }
  if (B.single_segment()) {
    B.settle();
  }
  // A probe within the single remaining segment must not reach the shared point.
  auto a_probe = [&](int* sa, int* pa, int* sb, int* pb) {
    A.probe(sa, pa, sb, pb);
  };
  auto b_probe = [&](int* sa, int* pa, int* sb, int* pb) {
    B.probe(sa, pa, sb, pb);
  };
  const double m = row.at(L[nl - 1], L[nl - 1].hi).x;
  while (!A.fixed() || !B.fixed()) {
    if (!A.fixed() && !B.fixed()) {
      int sa1, pa1, sa2, pa2, sb1, pb1, sb2, pb2;
      a_probe(&sa1, &pa1, &sa2, &pa2);
      b_probe(&sb1, &pb1, &sb2, &pb2);
      const Pt a1 = row.at(L[sa1], pa1), a2 = row.at(L[sa2], pa2);
      const Pt b1 = row.at(R[sb1], pb1), b2 = row.at(R[sb2], pb2);
      const Pt l{(a1.x + a2.x) / 2, (a1.y + a2.y) / 2}, r{(b1.x + b2.x) / 2, (b1.y + b2.y) / 2};
      if (slope_cmp(a1, a2, l, r) >= 0) {  // 1': the left end precedes alpha
        A.keep_before(sa1, pa1);
      } else if (slope_cmp(l, r, b1, b2) >= 0) {  // 2': the right end succeeds beta
        B.keep_after(sb2, pb2);
      } else if (gamma_x_cmp(a1, a2, b1, b2, m) <= 0) {  // 3': the left end succeeds alpha
        A.keep_after(sa2, pa2);
      } else {  // 3': the right end precedes beta
        B.keep_before(sb1, pb1);
      }
    } else if (!A.fixed()) {  // the tangent from R's vertex b to L
      int sa1, pa1, sa2, pa2, sb, pb;
      a_probe(&sa1, &pa1, &sa2, &pa2);
      B.first(&sb, &pb);
      const Pt a1 = row.at(L[sa1], pa1), a2 = row.at(L[sa2], pa2), b = row.at(R[sb], pb);
      if (slope_cmp(a1, a2, a1, b) < 0) {  // a2 below the chord a1 b: the tangent is past a1
        A.keep_after(sa2, pa2);
      } else {
        A.keep_before(sa1, pa1);
      }
    } else {  // the tangent from L's vertex a to R
      int sb1, pb1, sb2, pb2, sa, pa;
      b_probe(&sb1, &pb1, &sb2, &pb2);
      A.first(&sa, &pa);
      const Pt b1 = row.at(R[sb1], pb1), b2 = row.at(R[sb2], pb2), a = row.at(L[sa], pa);
      if (slope_cmp(b1, b2, a, b1) < 0) {  // b2 below the chord a b1: the tangent is past b1
        B.keep_after(sb2, pb2);
      } else {
        B.keep_before(sb1, pb1);
      }
    }
  }
  A.first(out_sa, out_pa);
  B.first(out_sb, out_pb);
}

template <bool W>
__device__ bool walk_capped(
    const Row<W>& row, Segs L, Segs R, int nr, int ls, int lp, int limit, int* sa, int* pa,
    int* sb, int* pb) {
  if (lp < 0) {
    --ls;
    lp = L[ls].hi;
  }
  int rs = 0, rp = R[0].lo + 1;
  if (rp > R[0].hi) {
    rs = 1;
    rp = R[1].lo;
  }
  Pt a = row.at(L[ls], lp), b = row.at(R[rs], rp);
  int steps = 0;
  bool moved = true;
  while (moved) {
    moved = false;
    while (rs < nr - 1 || rp < R[rs].hi) {
      int ns = rs, np = rp + 1;
      if (np > R[rs].hi) {
        ++ns;
        np = R[ns].lo;
      }
      const Pt q = row.at(R[ns], np);
      if (turn(a, b, q) > 0) {
        break;
      }
      rs = ns;
      rp = np;
      b = q;
      moved = true;
      if (++steps > limit) {
        return false;
      }
    }
    while (ls > 0 || lp > L[0].lo) {
      int ns = ls, np = lp - 1;
      if (np < L[ls].lo) {
        --ns;
        np = L[ns].hi;
      }
      const Pt q = row.at(L[ns], np);
      if (turn(q, a, b) > 0) {
        break;
      }
      ls = ns;
      lp = np;
      a = q;
      moved = true;
      if (++steps > limit) {
        return false;
      }
    }
  }
  *sa = ls;
  *pa = lp;
  *sb = rs;
  *pb = rp;
  return true;
}


template <bool W, bool WARP>
__device__ void merge_pair_b3(
    Hulls h, const Row<W>& row, int64_t first, int64_t half, int lane, int limit) {
  Seg* Lw = h.seg + first;
  const Segs L{Lw}, R{Lw + half};
  const int nl = h.nseg[first], nr = h.nseg[first + half];
  const Seg last = L[nl - 1], head = R[0];
  const Pt shared = row.at(last, last.hi);
  const Pt before = last.hi > last.lo ? row.at(last, last.hi - 1) : row.at(L[nl - 2], L[nl - 2].hi);
  const Pt after = head.hi > head.lo ? row.at(head, head.lo + 1) : row.at(R[1], R[1].lo);
  int keep, hi_l, from, lo_r;
  if (turn(before, shared, after) > 0) {
    keep = nl;
    hi_l = last.hi;
    from = 0;
    lo_r = head.lo + 1;
    if (lo_r > head.hi) {
      from = 1;
      lo_r = R[1].lo;
    }
  } else {
    int sa = 0, pa = 0, sb = 0, pb = 0;
    if constexpr (WARP) {
      // Few, long merges: the package's warp-parallel search hides the memory latency that
      // a sequential O(log) search by one lane would pay in full.
      bridge_start<W, true>(row, L.p, nl, R.p, nr, lane, &sa, &pa);
      tangent(row, R.p, nr, row.at(L[sa], pa), L.p, sa, pa, &sb, &pb);
    } else {
      if (!walk_capped(row, L, R, nr, nl - 1, last.hi - 1 >= last.lo ? last.hi - 1 : -1, limit,
                       &sa, &pa, &sb, &pb)) {
        bridge_three_cases(row, L, nl, R, nr, &sa, &pa, &sb, &pb);
      }
    }
    keep = sa + 1;
    hi_l = pa;
    from = sb;
    lo_r = pb;
  }
  const int moved = nr - from;
  if constexpr (WARP) {
    __syncwarp();
    if (lane == 0) {
      Lw[keep - 1].hi = hi_l;
    }
    for (int base = 0; base < moved; base += kLanes) {
      const int k = base + lane;
      Seg s{};
      if (k < moved) {
        s = R[from + k];
        if (k == 0) {
          s.lo = lo_r;
        }
      }
      __syncwarp();
      if (k < moved) {
        Lw[keep + k] = s;
      }
      __syncwarp();
    }
  } else {
    Lw[keep - 1].hi = hi_l;
    for (int k = 0; k < moved; ++k) {
      Seg s = R[from + k];
      if (k == 0) {
        s.lo = lo_r;
      }
      Lw[keep + k] = s;
    }
  }
  if (lane == 0) {
    h.nseg[first] = keep + moved;
  }
}

template <bool W, bool WARP>
__global__ void merge_b3_kernel(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row, int64_t half,
    int limit) {
  const int64_t id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                     (WARP ? kLanes : 1);
  const int lane = WARP ? threadIdx.x % kLanes : 0;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs;
  const Row<W> row{h, cy + r * (per_row + 1), W ? cx + r * (per_row + 1) : nullptr};
  merge_pair_b3<W, WARP>(h, row, r * per_row + id % pairs * 2 * half, half, lane, limit);
}

template <bool W>
void merge_rounds_b3(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
    cudaStream_t stream) {
  int level = 0;
  for (int64_t half = 1; half < per_row; half *= 2, ++level) {
    const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
    const int limit = 8 * (level + 10);  // T_r = c (r + log2 C), c = 8: a search step costs ~5 walk steps
    if (merges >= kThreadMerges) {
      merge_b3_kernel<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half, limit);
    } else {
      merge_b3_kernel<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half, limit);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// As the package's run(): as much shared memory as the SM allows, for the same occupancy.
template <typename scalar_t, bool W>
void carve() {
  static const bool carved = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(chunk_kernel<scalar_t, W, 1>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)carved;
}

at::Tensor isotonic(const at::Tensor& y, const c10::optional<at::Tensor>& w, bool increasing) {
  if (y.size(1) <= 4096) {
    return isotonic_tiles_cuda(y, w, increasing);  // one chunk per problem: nothing to merge
  }
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty_like(y);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "bridge3w", [&] {
    auto go = [&](auto weighted) {
      constexpr bool W = decltype(weighted)::value;
      const scalar_t* yp = y.data_ptr<scalar_t>();
      const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
      run_chunks<scalar_t, W, Chunk<scalar_t, W, 1>::kN>(
          y, W ? &*w : nullptr, increasing, out,
          [&](int64_t chunks, int64_t n, int64_t per_row, scalar_t sign, scalar_t* o, Hulls h,
              cudaStream_t stream) {
            carve<scalar_t, W>();
            chunk_kernel<scalar_t, W, 1><<<chunks, kLanes, 0, stream>>>(
                yp, wp, n, per_row, sign, o, h);
          },
          merge_rounds_b3<W>);
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_bridge3w

TORCH_LIBRARY(tiexp_bridge3w, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_bridge3w, CUDA, m) {
  m.impl("isotonic", &tiexp_bridge3w::isotonic);
}
