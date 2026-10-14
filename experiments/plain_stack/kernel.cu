// Ablation: the lane hulls by the textbook monotone chain - a stack of point indices and
// coordinates in local memory, a branchy pop loop - instead of lane_chain's bit mask with a
// branchless pop-or-push step. Only rows that go through chunk_kernel (n > 512); everything
// else is the package's.
#include "tiles.cuh"

// A named namespace: see experiments/README.md.
namespace tiexp_plain_stack {

using namespace torchisotonic;

template <bool W, typename acc_t>
__device__ __noinline__ acc_t plain_turn_slow(
    acc_t ax, acc_t ay, acc_t axl, acc_t ayl, acc_t bx, acc_t by, acc_t bxl, acc_t byl, acc_t qx,
    acc_t qy, acc_t qxl, acc_t qyl, acc_t dx1, acc_t dx2) {
  const Pair<acc_t> py1 = diff2(by, byl, ay, ayl), py2 = diff2(qy, qyl, ay, ayl);
  const Pair<acc_t> px1 = W ? diff2(bx, bxl, ax, axl) : Pair<acc_t>{dx1, acc_t(0)};
  const Pair<acc_t> px2 = W ? diff2(qx, qxl, ax, axl) : Pair<acc_t>{dx2, acc_t(0)};
  return cross2(px1, py1, px2, py2);
}

// PAIRS as in lane_chain: the sums come as hi + lo and the predicate uses both parts.
template <bool W, bool PAIRS, typename acc_t>
__device__ uint32_t plain_chain(
    const acc_t* sy, const int16_t* syl, const acc_t* sx, const int16_t* sxl, int count,
    acc_t by_, acc_t bx_) {
  int si[kLanes + 1];
  acc_t stx[kLanes + 1], sty[kLanes + 1], stxl[kLanes + 1], styl[kLanes + 1];
  int h = 0;
  si[h] = 0;
  stx[h] = 0;
  sty[h] = 0;
  stxl[h] = 0;
  styl[h] = 0;
  ++h;
  for (int k = 1; k <= count; ++k) {
    const acc_t qx = W ? sx[k - 1] : acc_t(k), qy = sy[k - 1];
    const acc_t qxl = W && PAIRS ? unpack_lo(qx, sxl[k - 1]) : acc_t(0);
    const acc_t qyl = PAIRS ? unpack_lo(qy, syl[k - 1]) : acc_t(0);
    while (h >= 2) {
      const acc_t ax = stx[h - 2], ay = sty[h - 2], bx = stx[h - 1], by = sty[h - 1];
      const acc_t axl = stxl[h - 2], ayl = styl[h - 2], bxl = stxl[h - 1], byl = styl[h - 1];
      const acc_t dx1 = bx - ax, dy1 = by - ay, dx2 = qx - ax, dy2 = qy - ay;
      const acc_t p1 = dx1 * dy2, p2 = dy1 * dx2;
      acc_t c = p1 - p2;
      if constexpr (PAIRS) {  // the package's filter and pair fallback (lane_chain)
        acc_t bound = fma(by_, fabs(dx1) + fabs(dx2), bound_p<acc_t>() * (fabs(p1) + fabs(p2)));
        if (W) {
          bound = fma(bx_, fabs(dy1) + fabs(dy2), bound);
        }
        if (!(fabs(c) > bound)) {
          c = plain_turn_slow<W>(ax, ay, axl, ayl, bx, by, bxl, byl, qx, qy, qxl, qyl, dx1, dx2);
        }
      }
      if (c > 0) {
        break;
      }
      --h;
    }
    si[h] = k;
    stx[h] = qx;
    sty[h] = qy;
    stxl[h] = qxl;
    styl[h] = qyl;
    ++h;
  }
  uint32_t mask = 0;
  for (int i = 0; i < h; ++i) {
    if (si[i] < count) {
      mask |= 1u << si[i];
    }
  }
  return mask;
}

// chunk_kernel, with plain_chain for lane_chain.
template <typename scalar_t, bool W, int WARPS>
__global__ void __launch_bounds__(WARPS * kLanes) stack_chunk_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ w,
    int64_t n,
    int64_t per_row,
    scalar_t sign,
    scalar_t* __restrict__ out,
    Hulls h) {
  using acc_t = scalar_t;
  using C = Chunk<acc_t, W, WARPS>;
  constexpr int kL = C::kL, kN = C::kN;
  __shared__ C c;
  const int t = threadIdx.x, lane = t % kLanes, warp = t / kLanes;  // t: this thread's lane
#ifdef TI_PHASES
  long long mark = clock64();
#endif
  const int64_t g = blockIdx.x, r = g / per_row, j = g % per_row, start = r * n + j * kN;
  const int len = static_cast<int>(n - j * kN < kN ? n - j * kN : kN);

  // Coalesced, and all 32 loads of a thread issued before any is used.
  acc_t ry[kLanes], rw[W ? kLanes : 1];
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kL + t;
    ry[i] = e < len ? static_cast<acc_t>(y[start + e]) : acc_t(0);
    if (W) {
      rw[i] = e < len ? static_cast<acc_t>(w[start + e]) : acc_t(0);
    }
  }
  bool nan = false;
#pragma unroll
  for (int i = 0; i < kLanes; ++i) {
    const int e = i * kL + t;
    const acc_t v = sign * ry[i];
    nan |= v != v;
    if (W) {
      c.sy[pad(e)] = rw[i] * v;
      c.sx[pad(e)] = rw[i];
    } else {
      c.sy[pad(e)] = v;
    }
  }
  nan = __syncthreads_or(nan);
  TI_MARK(0);
  TI_STOP(0, __float_as_int(static_cast<float>(c.sy[pad(t)])) + int(nan));

  // Each lane: the hull of its points 0..count by the monotone chain. The stack is the set
  // bits of mask (points the lane owns: 0..count-1; point count is the next lane's 0) plus
  // its top b; a is the entry under b. Every iteration either pops b or pushes the next point
  // q, chosen by selects rather than branches: lanes stay together and the loop body stays
  // short (the kernel is bound by instructions issued, not by latency).
  const int count = imax(0, imin(kLanes, len - t * kLanes));
  uint32_t mask = count > 0 ? 1u : 0u;
  acc_t tx = 0, ty = 0, txl = 0, tyl = 0, my = 0, mx = 0;
  {
    const int base = t * (kLanes + 1);
    acc_t py[kLanes], px[W ? kLanes : 1];
#pragma unroll
    for (int k = 0; k < kLanes; ++k) {
      py[k] = c.sy[base + k];
      if (W) {
        px[k] = c.sx[base + k];
      }
    }
#pragma unroll
    for (int k = 0; k < kLanes; ++k) {  // py[k]: point k + 1
      scan_step<C::kPairs>(ty, tyl, k < count ? py[k] : acc_t(0));
      c.sy[base + k] = ty;
      my = fmax(my, fabs(ty));
      if (C::kPairs) {
        c.syl[base + k] = pack_lo(ty, tyl);
      }
      if (W) {
        scan_step<C::kPairs>(tx, txl, k < count ? px[k] : acc_t(0));
        c.sx[base + k] = tx;
        mx = fmax(mx, fabs(tx));
        if (C::kPairs) {
          c.sxl[base + k] = pack_lo(tx, txl);
        }
      }
    }
    if (!W) {
      tx = acc_t(count);
    }
    mask = plain_chain<W, C::kPairs>(c.sy + base, c.syl + (C::kPairs ? base : 0),
                                     c.sx + (W ? base : 0),
                                     c.sxl + (W && C::kPairs ? base : 0), count,
                                     bound_m<acc_t>() * my, bound_m<acc_t>() * mx);
  }

  TI_MARK(1);
  TI_STOP(1, static_cast<int>(mask) ^ __float_as_int(static_cast<float>(ty)));
  const double iy = lane_starts<acc_t, WARPS>(ty, tyl, lane, warp, c.by, c.wy);
  double ix = 0.0;
  if constexpr (W) {
    ix = lane_starts<acc_t, WARPS>(tx, txl, lane, warp, c.bx, c.wx);
  }

  TI_MARK(2);
  TI_STOP(2, static_cast<int>(mask) ^ __float_as_int(static_cast<float>(c.by[0][t])));
  // Merge the lane hulls: round `level` merges pairs of 2^level lanes. Merge q joins lanes
  // [t0, t0 + half) and [t0 + half, t0 + 2 half); its first lane walks to the bridge, and its
  // lanes clear the mask bits under it. Up to round 4 a merge is within a warp.
  const int active = (len + kLanes - 1) / kLanes;
  for (int level = 0; (1 << level) < active; ++level) {
    const bool in_warp = level < 5;
    const int half = 1 << level, G = 2 * half;
    const int t0 = t / G * G, tm = t0 + half;
    const bool valid = tm < active;
    c.mask[t] = mask;
    if (in_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
    int a = -1, b = -1, steps = 0;
    if (valid && t == t0) {
      const int t1 = imin(t0 + G, active);
      steps = walk_bridge(c, [&](int p) { return chunk_point(c, p); }, t0 * kLanes, tm * kLanes,
                          t1, t1 < active ? t1 * kLanes : len, &a, &b);
    }
    if (in_warp) {
      a = __shfl_sync(kFull, a, t0 % kLanes);
      b = __shfl_sync(kFull, b, t0 % kLanes);
    } else {
      if (valid && t == t0) {
        c.bridge[0][t0 / G] = a;
        c.bridge[1][t0 / G] = b;
      }
      __syncthreads();
      if (valid) {
        a = c.bridge[0][t0 / G];
        b = c.bridge[1][t0 / G];
      }
    }
#ifdef TI_PHASES
    {  // walk steps: in all, and the longest of the round (what the warp waits for)
      int most = steps, all = steps;
      for (int d = 16; d > 0; d >>= 1) {
        most = imax(most, __shfl_xor_sync(kFull, most, d));
        all += __shfl_xor_sync(kFull, all, d);
      }
      if (lane == 0) {
        atomicAdd(&phase_cycles[5], static_cast<unsigned long long>(all));
        atomicAdd(&phase_cycles[6], static_cast<unsigned long long>(most));
      }
    }
#else
    (void)steps;
#endif
    if (valid && a >= 0) {
      mask &= ~between(a, b, t);
    }
    if (in_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
  }

  // Rank the vertices: lane by lane, then the chunk's end point (point len) at rank top.
  c.mask[t] = mask;
  const int count_v = __popc(mask);
  const int incl = warp_inclusive_sum(count_v, lane);
  if (lane == kLanes - 1) {
    c.wi[warp] = incl;
  }
  __syncthreads();
  int first = incl - count_v;
  for (int v = 0; v < warp; ++v) {
    first += c.wi[v];
  }
  c.off[t] = first;
  if (t == kL - 1) {
    c.off[kL] = first + count_v;
  }
  __syncthreads();
  TI_MARK(3);
  TI_STOP(3, static_cast<int>(mask) ^ c.off[kL]);
  const int top = c.off[kL];

  if (per_row == 1) {  // the chunk is the whole problem: write its fit
    // Each lane its own 32 elements: backwards, the slope of the edge out of each of its
    // vertices (to the next vertex: its own, or the first one of a later lane, or the end);
    // forwards, every element takes the slope of the last vertex at or before it (for the
    // elements before its first vertex, the last slope of an earlier lane). No popc, no ffs,
    // a division per vertex only. The slopes are staged where the sums were, then written
    // coalesced.
    const int p0 = t * kLanes;
    int after = mask ? p0 + __ffs(mask) - 1 : len;  // suffix minimum of first vertices
    for (int d = 1; d < kLanes; d <<= 1) {
      const int o = __shfl_down_sync(kFull, after, d);
      if (lane + d < kLanes && o < after) {
        after = o;
      }
    }
    if (lane == 0) {
      c.wi[warp] = after;  // the warp's first vertex
    }
    int next_first = __shfl_down_sync(kFull, after, 1);
    __syncthreads();
    if (lane == kLanes - 1) {
      next_first = len;
    }
    for (int v = warp + 1; v < WARPS; ++v) {
      next_first = imin(next_first, c.wi[v]);
    }
    next_first = imin(next_first, len);
    acc_t slope[kLanes];
    acc_t last = 0;  // the slope out of this lane's last vertex
    {
      bool found = false;
      int nv = next_first;  // the vertex after the one being looked at
      Local<acc_t> pn = local(c, nv);
#pragma unroll
      for (int k = kLanes - 1; k >= 0; --k) {
        if ((mask >> k) & 1u) {
          const Local<acc_t> pa = local(c, p0 + k);
          Pair<acc_t> dx, dy;
          if (nv / kLanes == t) {  // both ends in this lane: no lane starts involved
            dy = diff2(pn.y, pn.yl, pa.y, pa.yl);
            dx = W ? diff2(pn.x, pn.xl, pa.x, pa.xl)
                   : Pair<acc_t>{to_acc<acc_t>(nv - p0 - k), acc_t(0)};
          } else {
            delta(c, pa, pn, dx, dy);
          }
          slope[k] = sign * ((dy.h + dy.l) / (dx.h + dx.l));
          if (!found) {
            last = slope[k];
            found = true;
          }
          nv = p0 + k;
          pn = pa;
        }
      }
    }
    // The slope coming into this lane: the last slope of the nearest earlier lane with
    // vertices (lane 0 always has one, its point 0).
    bool has = mask != 0;
    acc_t into = last;
    for (int d = 1; d < kLanes; d <<= 1) {
      const acc_t o = __shfl_up_sync(kFull, into, d);
      const bool oh = __shfl_up_sync(kFull, has, d);
      if (lane >= d && !has) {
        into = o;
        has = oh;
      }
    }
    acc_t prev = __shfl_up_sync(kFull, into, 1);
    bool prev_has = __shfl_up_sync(kFull, has, 1) && lane > 0;
    __syncthreads();  // everyone has read c.wi
    if (lane == kLanes - 1) {
      c.wi[warp] = has;
      c.wf[warp] = into;
    }
    __syncthreads();
    for (int v = warp - 1; v >= 0 && !prev_has; --v) {
      if (c.wi[v]) {
        prev = c.wf[v];
        prev_has = true;
      }
    }
    {
      acc_t cur = prev;
#pragma unroll
      for (int k = 0; k < kLanes; ++k) {
        if ((mask >> k) & 1u) {
          cur = slope[k];
        }
        c.sy[p0 + t + k] = cur;  // = pad(p0 + k)
      }
    }
    __syncthreads();
    for (int e = t; e < len; e += kL) {
      out[start + e] = nan ? std::numeric_limits<scalar_t>::quiet_NaN() : c.sy[pad(e)];
    }
    TI_MARK(4);
    return;
  }

  // Otherwise write the chunk's hull: every lane its vertices, the last lane the end point.
  const int64_t voff = g * (kN + 1);
  auto put = [&](int i, int p) {
    const Local<acc_t> q = local(c, p);
    const int u = p / kLanes;
    h.p[voff + i] = static_cast<int32_t>(j * kN + p);
    h.y[voff + i] = static_cast<double>(c.by[0][u]) + static_cast<double>(c.by[1][u]) +
                    static_cast<double>(q.y) + static_cast<double>(q.yl);
    if (W) {
      h.x[voff + i] = static_cast<double>(c.bx[0][u]) + static_cast<double>(c.bx[1][u]) +
                      static_cast<double>(q.x) + static_cast<double>(q.xl);
    }
  };
  {
    int i = first;
    for (uint32_t m = mask; m; m &= m - 1) {
      put(i++, t * kLanes + __ffs(m) - 1);
    }
  }
  if (t == kL - 1) {
    put(top, len);
    h.seg[g] = Seg{voff, static_cast<int32_t>(j), 0, top, 0};
    h.nseg[g] = 1;
    h.total_y[g] = iy;
    if (W) {
      h.total_x[g] = ix;
    }
    if (nan) {
      h.row_nan[r] = 1;
    }
  }
  TI_MARK(4);
}


// As the package's run(): as much shared memory as the SM allows, for the same occupancy.
template <typename scalar_t, bool W, typename Kernel>
void carve(Kernel kernel) {
  static const bool carved = [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)carved;
}

at::Tensor isotonic(const at::Tensor& y, const c10::optional<at::Tensor>& w, bool increasing) {
  if (y.size(1) <= 512) {
    return isotonic_tiles_cuda(y, w, increasing);  // packed rows: not under test
  }
  const c10::cuda::CUDAGuard guard(y.device());
  at::Tensor out = at::empty_like(y);
  AT_DISPATCH_FLOATING_TYPES(y.scalar_type(), "plain_stack", [&] {
    auto go = [&](auto weighted) {
      constexpr bool W = decltype(weighted)::value;
      const scalar_t* yp = y.data_ptr<scalar_t>();
      const scalar_t* wp = W ? w->data_ptr<scalar_t>() : nullptr;
      const int64_t n = y.size(1);
      auto launch = [&](auto warps_c) {
        constexpr int warps = decltype(warps_c)::value;
        run_chunks<scalar_t, W, Chunk<scalar_t, W, warps>::kN>(
            y, W ? &*w : nullptr, increasing, out,
            [&](int64_t chunks, int64_t nn, int64_t per_row, scalar_t sign, scalar_t* o,
                Hulls h, cudaStream_t stream) {
              carve<scalar_t, W>(stack_chunk_kernel<scalar_t, W, warps>);
              stack_chunk_kernel<scalar_t, W, warps><<<chunks, warps * kLanes, 0, stream>>>(
                  yp, wp, nn, per_row, sign, o, h);
            });
      };
      // The package's chunk sizes: one chunk of as few warps as cover a row of up to 4096
      // (as many as shared memory allows), one-warp chunks beyond.
      constexpr int most = chunk_warps<scalar_t, W>();
      if (n <= 1024 || n > 4096 || most < 2) {
        launch(std::integral_constant<int, 1>{});
      } else if (n <= 2048 || most < 4) {
        launch(std::integral_constant<int, (most >= 2 ? 2 : 1)>{});
      } else {
        launch(std::integral_constant<int, (most >= 4 ? 4 : 1)>{});
      }
    };
    if (w.has_value()) {
      go(std::true_type{});
    } else {
      go(std::false_type{});
    }
  });
  return out;
}

}  // namespace tiexp_plain_stack

TORCH_LIBRARY(tiexp_plain_stack, m) {
  m.def("isotonic(Tensor y, Tensor? w, bool increasing) -> Tensor");
}

TORCH_LIBRARY_IMPL(tiexp_plain_stack, CUDA, m) {
  m.impl("isotonic", &tiexp_plain_stack::isotonic);
}
