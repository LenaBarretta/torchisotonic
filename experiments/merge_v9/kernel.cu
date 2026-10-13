// The merge code of commit 430a916 (before the precision work: Pt{x, y}, absolute
// coordinates, Row without lo parts), on today's chunk and fit kernels. A control: if it is
// as fast as the package in the same run, the merge stage's 0.47 -> 0.70 ms (random 1e7) was
// not the code; if it is faster, the precision work costs that much and must be made cheaper.
#include "tiles.cuh"

namespace tiexp_merge_v9 {

using namespace torchisotonic;

struct Pt9 {
  double x, y;
};

__device__ __forceinline__ double turn9(const Pt9& a, const Pt9& b, const Pt9& d) {
  return (b.x - a.x) * (d.y - a.y) - (b.y - a.y) * (d.x - a.x);
}

template <bool W>
struct Row9 {  // one problem's vertices, in its coordinates
  Hulls h;
  const double* cy;  // chunk starts
  const double* cx;

  __device__ __forceinline__ Pt9 at(const Seg& s, int pos) const {
    const int64_t i = s.voff + pos;
    return Pt9{W ? cx[s.chunk] + h.x[i] : static_cast<double>(h.p[i]), cy[s.chunk] + h.y[i]};
  }
};

template <bool W>
__device__ void tangent9(const Row9<W>& row, const Seg* R, int nr, const Pt9& p, int* ts, int* tp) {
  // Slopes from p fall, then rise, along R. First narrow it down by segments' first vertices
  // to (first of segment s - 1, first of segment s + 1], then search that window.
  int sa = 0, pa = R[0].lo + 1, na = R[0].hi - R[0].lo;  // window part A: after a first vertex
  int sb = 0, pb = 0, nb = 0;                            // part B: a whole segment
  int sc = 0, pc = 0, nc = 0;                            // part C: the next first vertex
  if (nr > 1) {
    int lo = 1, hi = nr - 1;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      const Seg u = R[mid], v = R[mid + 1];
      if (turn9(p, row.at(u, u.lo), row.at(v, v.lo)) < 0) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const Seg a = R[lo - 1], b = R[lo];
    sa = lo - 1;
    pa = a.lo + 1;
    na = a.hi - a.lo;
    sb = lo;
    pb = b.lo;
    nb = b.hi - b.lo + 1;
    if (lo + 1 < nr) {
      sc = lo + 1;
      pc = R[lo + 1].lo;
      nc = 1;
    }
  }
  auto vertex = [&](int i, int* s, int* pos) {
    if (i < na) {
      *s = sa;
      *pos = pa + i;
    } else if (i - na < nb) {
      *s = sb;
      *pos = pb + i - na;
    } else {
      *s = sc;
      *pos = pc;
    }
  };
  int lo = 0, hi = na + nb + nc - 1;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    int s0, p0, s1, p1;
    vertex(mid, &s0, &p0);
    vertex(mid + 1, &s1, &p1);
    if (turn9(p, row.at(R[s0], p0), row.at(R[s1], p1)) < 0) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  vertex(lo, ts, tp);
}

// The left end of the bridge between L and R (sharing L's last vertex and R's first), when
// there is one: the first vertex of L whose successor is not below its tangent9 to R.
template <bool W, bool WARP>
__device__ void bridge_start9(
    const Row9<W>& row, const Seg* L, int nl, const Seg* R, int nr, int lane, int* ss, int* sp) {
  auto further = [&](int s, int pos) {  // the bridge starts right of vertex (s, pos)
    const Seg cs = L[s];
    const Pt9 p = row.at(cs, pos);
    const Pt9 next = pos < cs.hi ? row.at(cs, pos + 1) : row.at(L[s + 1], L[s + 1].lo);
    int ts, tp;
    tangent9(row, R, nr, p, &ts, &tp);
    return turn9(p, row.at(R[ts], tp), next) < 0;
  };
  // Candidates: every vertex but the shared one. By segments' first vertices first...
  const int firsts = L[nl - 1].lo == L[nl - 1].hi ? nl - 1 : nl;
  const int s = first_false<WARP>(firsts, lane, [&](int i) { return further(i, L[i].lo); });
  if (s == 0) {
    *ss = 0;
    *sp = L[0].lo;
    return;
  }
  // ...then within (first of segment s - 1, first of segment s].
  const Seg a = L[s - 1];
  const int inside = (s - 1 == nl - 1 ? a.hi - 1 : a.hi) - a.lo;
  const int width = inside + (s < firsts ? 1 : 0);
  if (width == 0) {
    *ss = s - 1;
    *sp = a.lo;
    return;
  }
  auto vertex = [&](int i, int* vs, int* vp) {
    if (i < inside) {
      *vs = s - 1;
      *vp = a.lo + 1 + i;
    } else {
      *vs = s;
      *vp = L[s].lo;
    }
  };
  const int i = first_false<WARP>(width, lane, [&](int i) {
    int vs, vp;
    vertex(i, &vs, &vp);
    return further(vs, vp);
  });
  vertex(i < width ? i : width - 1, ss, sp);
}

constexpr int kSegWalk9 = 64;  // steps of walk_segments9 before it gives up

// The bridge between L and R (sharing L's last vertex and R's first), found by walking from
// the shared point as in the chunks (walk_bridge), over the segment lists: few steps on the
// short hulls of the many-merge rounds. False (nothing written) after kSegWalk9 steps, for a
// search instead. One thread. (ls, lp): L's vertex before the shared one, lp = -1 meaning
// the last vertex of the segment before ls.
template <bool W>
__device__ bool walk_segments9(
    const Row9<W>& row, const Seg* L, const Seg* R, int nr, int ls, int lp, int* sa, int* pa,
    int* sb, int* pb) {
  if (lp < 0) {
    --ls;
    lp = L[ls].hi;
  }
  int rs = 0, rp = R[0].lo + 1;  // R's vertex after the shared one
  if (rp > R[0].hi) {
    rs = 1;
    rp = R[1].lo;
  }
  Pt9 a = row.at(L[ls], lp), b = row.at(R[rs], rp);
  int steps = 0;
  bool moved = true;
  while (moved) {
    moved = false;
    while (rs < nr - 1 || rp < R[rs].hi) {  // b is not R's last vertex
      int ns = rs, np = rp + 1;
      if (np > R[rs].hi) {
        ++ns;
        np = R[ns].lo;
      }
      const Pt9 q = row.at(R[ns], np);
      if (turn9(a, b, q) > 0) {
        break;
      }
      rs = ns;
      rp = np;
      b = q;
      moved = true;
      if (++steps > kSegWalk9) {
        return false;
      }
    }
    while (ls > 0 || lp > L[0].lo) {  // a is not L's first vertex
      int ns = ls, np = lp - 1;
      if (np < L[ls].lo) {
        --ns;
        np = L[ns].hi;
      }
      const Pt9 q = row.at(L[ns], np);
      if (turn9(q, a, b) > 0) {
        break;
      }
      ls = ns;
      lp = np;
      a = q;
      moved = true;
      if (++steps > kSegWalk9) {
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

// Pairs of neighbouring nodes (half chunks wide) of one problem merge, one warp per pair
// (WARP: few pairs, long lists) or one thread per pair (many pairs, short lists: the warp's
// 32 parallel candidates would be mostly wasted float64 work). A node's segment list starts
// at its first chunk's slot; the merged list goes where the left one was.
template <bool W, bool WARP>
__device__ void merge_pair9(Hulls h, const Row9<W>& row, int64_t first, int64_t half, int lane) {
  Seg* L = h.seg + first;
  const Seg* R = L + half;
  const int nl = h.nseg[first], nr = h.nseg[first + half];

  const Seg last = L[nl - 1], head = R[0];
  const Pt9 shared = row.at(last, last.hi);
  const Pt9 before = last.hi > last.lo ? row.at(last, last.hi - 1) : row.at(L[nl - 2], L[nl - 2].hi);
  const Pt9 after = head.hi > head.lo ? row.at(head, head.lo + 1) : row.at(R[1], R[1].lo);
  int keep, hi_l, from, lo_r;  // L keeps segments [0, keep), the last up to hi_l; then R's from `from`
  if (turn9(before, shared, after) > 0) {  // convex at the shared point: concatenate
    keep = nl;
    hi_l = last.hi;
    from = 0;
    lo_r = head.lo + 1;  // the shared point once
    if (lo_r > head.hi) {
      from = 1;
      lo_r = R[1].lo;
    }
  } else {
    int sa, pa, sb, pb;
    if (WARP || !walk_segments9(row, L, R, nr, nl - 1, last.hi - 1 >= last.lo ? last.hi - 1 : -1,
                               &sa, &pa, &sb, &pb)) {
      bridge_start9<W, WARP>(row, L, nl, R, nr, lane, &sa, &pa);
      tangent9(row, R, nr, row.at(L[sa], pa), &sb, &pb);
    }
    keep = sa + 1;
    hi_l = pa;
    from = sb;
    lo_r = pb;
  }
  // Move R's tail down to follow L's. Destinations never pass their sources, so reading a
  // round of 32 before writing it is enough.
  const int moved = nr - from;
  if constexpr (WARP) {
    __syncwarp();
    if (lane == 0) {
      L[keep - 1].hi = hi_l;
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
        L[keep + k] = s;
      }
      __syncwarp();
    }
  } else {
    L[keep - 1].hi = hi_l;
    for (int k = 0; k < moved; ++k) {
      Seg s = R[from + k];
      if (k == 0) {
        s.lo = lo_r;
      }
      L[keep + k] = s;
    }
  }
  if (lane == 0) {
    h.nseg[first] = keep + moved;
  }
}

template <bool W, bool WARP>
__global__ void merge_kernel9(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row, int64_t half) {
  const int64_t id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                     (WARP ? kLanes : 1);
  const int lane = WARP ? threadIdx.x % kLanes : 0;
  const int64_t pairs = (per_row + half - 1) / (2 * half);
  if (id >= batch * pairs) {
    return;
  }
  const int64_t r = id / pairs;
  const Row9<W> row{h, cy + r * (per_row + 1), W ? cx + r * (per_row + 1) : nullptr};
  merge_pair9<W, WARP>(h, row, r * per_row + id % pairs * 2 * half, half, lane);
}


template <bool W>
void merge_rounds9(
    Hulls h, const double* cy, const double* cx, int64_t batch, int64_t per_row,
    cudaStream_t stream) {
  for (int64_t half = 1; half < per_row; half *= 2) {
    const int64_t merges = batch * ((per_row + half - 1) / (2 * half));
    if (merges >= kThreadMerges) {
      merge_kernel9<W, false><<<(merges + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
    } else {
      merge_kernel9<W, true><<<blocks_for_warps(merges), kThreads, 0, stream>>>(
          h, cy, cx, batch, per_row, half);
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
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "merge_v9", [&] {
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
          merge_rounds9<W>);
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_merge_v9

TORCH_LIBRARY(tiexp_merge_v9, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_merge_v9, CUDA, m) {
  m.impl("isotonic", &tiexp_merge_v9::isotonic);
}
