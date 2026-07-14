# Bugs Found in Upstream HAL (Java)

Running log of defects found in the HAL Java source (`HAL/...`). Each is a
genuine bug in the upstream repo, recorded here so they can be reported
back to the HAL maintainers.

For each entry: the file/method, what the code actually does, and why
that's wrong.

---

## 1. `Util.java` — `GetBit(int, int)` / `GetBit(long, int)` ignore the bit index

**Location:** `HAL/Util.java`, `GetBit(int v, int i)` and `GetBit(long v, int i)`

```java
public static boolean GetBit(int v, int i) {
    return v >> i % 1 != 0;
}
```

Due to operator precedence, this parses as `v >> (i % 1)`. Since `i % 1` is
always `0` for any integer `i`, the shift amount is always zero — the
function ignores `i` entirely and just returns `v != 0`. The evident intent
was `((v >> i) & 1) != 0`.

---

## 2. `Util.java` — `FlipBit(long, int)` truncates for bit index >= 32

**Location:** `HAL/Util.java`, `FlipBit(long v, int i)`

```java
public static long FlipBit(long v, int i) {
    return v ^ (1 << i);
}
```

`1 << i` is evaluated as a 32-bit `int` shift before being widened to
`long`. Java masks int shift amounts to their low 5 bits, so for `i >= 32`
this silently wraps (e.g. `i=32` behaves like `i=0`) instead of flipping the
intended high bit. Since this overload exists specifically to support bit
indices beyond 31, the bug defeats its own purpose for roughly half its
valid input range. The fix is a 64-bit shift (`1L << i`).

---

## 3. `Util.java` — `HeatMapRBG(val, min, max)` calls the wrong single-argument variant

**Location:** `HAL/Util.java`, `HeatMapRBG(double val, double min, double max)`

```java
public static int HeatMapRBG(double val, double min, double max) {
    val = Scale0to1(val, min, max);
    return HeatMapBGR(val);   // should be HeatMapRBG(val)
}
```

Every sibling min/max-overload (`HeatMapRGB`, `HeatMapGRB`, `HeatMapGBR`,
`HeatMapBRG`, `HeatMapBGR`) rescales `val` and calls its own matching
single-argument palette function. `HeatMapRBG`'s min/max overload is the
only one that calls a *different* palette (`HeatMapBGR`) instead of its own
— almost certainly a copy-paste error, since it means
`HeatMapRBG(val, min, max)` and `HeatMapRBG(val)` produce different
palettes for the same rescaled input.

---

## 4. `Interfaces/Grid2D.java` and `Interfaces/Grid3D.java` — `WrapI` wraps `y` using `Xdim()` instead of `Ydim()`

**Location:** `HAL/Interfaces/Grid2D.java`, `WrapI(int x, int y)`
(and the analogous `HAL/Interfaces/Grid3D.java`, `WrapI(int x, int y, int z)`)

```java
default public int WrapI(int x, int y) {
    if (In(x, y)) return I(x, y);
    if (IsWrapX()) { x = Util.Wrap(x, Xdim()); }
    if (IsWrapY()) { y = Util.Wrap(y, Xdim()); }  // should be Ydim()
    if (In(x, y)) return I(x, y);
    throw new IllegalArgumentException("cannot map to index in bounds!");
}
```

`y` is wrapped modulo `Xdim()` instead of `Ydim()`. On any non-square grid
(`xDim != yDim`) with `y` out of bounds and `wrapY` enabled, this produces
an incorrect wrapped index (or fails to bring `y` into bounds at all,
hitting the `throw`).

---

## 5. `Interfaces/Grid3D.java` — `DispZ` checks `IsWrapY()` instead of `IsWrapZ()`

**Location:** `HAL/Interfaces/Grid3D.java`, `DispZ(double z1, double z2)`

```java
default public double DispZ(double z1, double z2) {
    if (IsWrapY()) {                      // should be IsWrapZ()
        return Util.DispWrap(z2, z1, Zdim());
    } else {
        return z2 - z1;
    }
}
```

The wraparound-displacement helper for the Z axis is gated on the Y-axis
wrap flag instead of the Z-axis one. On any grid where `wrapY != wrapZ`,
`DispZ` (and everything built on it — `DistSquared`, `Dist`) computes the
wrong Z displacement.

---

## 6. `Interfaces/Grid3D.java` — `GetHoodI` assigns the wrapped `z` into `y`

**Location:** `HAL/Interfaces/Grid3D.java`, `GetHoodI(int[] hood, int centerX, int centerY, int centerZ, int entryIndex)`

```java
if (!Util.InDim(z, Zdim())) {
    if (IsWrapZ()) {
        y = Util.Wrap(z, Zdim());   // should assign to z, not y
    } else {
        return -1;
    }
}
return I(x, y, z);
```

When `z` is out of bounds and wraps, the wrapped value is written into `y`
instead of `z`. This corrupts `y` (overwriting whatever value it already
held) while leaving `z` unwrapped and still out of bounds, so `I(x, y, z)`
returns a corrupted index built from the wrong `y` and an invalid `z`.

---

## 7. `PopulationGridBase.java` (and `PopulationGridLongBase.java`) — `SetupSparseIndices` double-counts `occupiedArea` on every dense→sparse transition

**Location:** `HAL/GridsAndAgents/PopulationGridBase.java`, `Update()` /
`SetupSparseIndices()` / `AddPopIndex()`

```java
// Update(), dense path:
for (int i = 0; i < agents.length; i++) {
    ...
    if (prev == 0 && agents[i] != 0) {
        occupiedArea++;             // (1) already tallies every occupied cell
    } else if (prev != 0 && agents[i] == 0) {
        occupiedArea--;
    }
}
if (!usingSparseIndices && occupiedArea <= length * 0.001) {
    usingSparseIndices = true;
    SetupSparseIndices();           // (2) re-scans and re-counts the same cells
}

// SetupSparseIndices():
for (int i = 0; i < length; i++) {
    if (agents[i] > 0) {
        AddPopIndex(i);              // increments occupiedArea AGAIN
    }
}

// AddPopIndex():
private void AddPopIndex(int i) {
    occupiedArea++;                  // no way to rebuild the linked list
    ...                              // without also bumping the counter
}
```

The dense-path loop in `Update()` already computes the correct
`occupiedArea` by incrementing/decrementing it as cells transition to/from
zero. When occupancy crosses the sparse threshold, `SetupSparseIndices()`
re-scans every cell to rebuild the `nextLiveIndex`/`prevLiveIndex` linked
list — but it does so by calling `AddPopIndex()`, which *also* increments
`occupiedArea` for every currently-occupied cell it links. Since those
cells were already counted by the dense-path loop moments earlier, this
inflates `occupiedArea` by however many cells are occupied at the moment
of the transition (not a flat off-by-one — the error scales with
occupancy). The same corrupted `occupiedArea` then persists and can
compound further on any later transition back to sparse mode (e.g. after
occupancy drops, rises, and falls again).

The fix is to rebuild the linked list in `SetupSparseIndices()` with a
separate link-only helper that does not touch `occupiedArea`, leaving the
counter as the dense-path loop (or prior sparse-path increments) already
correctly left it. `AddPopIndex` (used only by the sparse-mode incremental
path in `Update()`, where a cell is transitioning for the first time) can
keep incrementing the counter, since that path only calls it for cells
that weren't already counted.

---

## 8. `Tools/Internal/PDEequations.java` — ~160 lines of abandoned, non-working 3D ADI code left in the file, fully commented out

**Location:** `HAL/Tools/Internal/PDEequations.java`, lines 843–1002

```java
//THE BELOW CODE IS A 3D ADI IMPLEMENTATION THAT SO FAR IS NOT WORKING!
//    //only changes from 2D to 3D are 4x diffRate for db, front and back added to d side of equation, and array accesses are different
//    public static void TDMA3x(final double[] in, final double[] out,final double[] scratch,final int xDim, final int yDim,final int zDim, final int iRow1,final int iRow2, final double diffRate) {
//        ...
```

This is a separate, self-admittedly-broken attempt at a 3D ADI (Alternating
Direction Implicit) diffusion solver — `TDMA3x`/`TDMA3y`/`TDMA3z` plus a
dispatching helper, all commented out, with the author's own header comment
stating it doesn't work. It has no callers (it's inert, commented-out
source) and is easy to mistake for the codebase's real 3D ADI support,
which actually lives elsewhere (`Tools/Internal/ADIequations.java`'s
`Diffusion3DADI`, called from `PDEGrid3D.DiffusionADI` — that code is live,
correct, and unrelated to this abandoned attempt). Worth deleting outright
rather than leaving in the file: it's not a reference implementation
(it's admittedly broken), it long predates the working version, and its
presence risks a future reader assuming it's the codebase's 3D ADI
solution when it isn't.

---

## 9. `Tools/ODESolver/ODESolver.java` — leftover `System.out.println` debug statement in `Runge45internal`

**Location:** `HAL/Tools/ODESolver/ODESolver.java`, in `Runge45internal` (just after the RK4/RK5 accumulation loop)

```java
System.out.println(s0[0]);
```

Every RKF45 step prints the first component of the intermediate RK4 state
to stdout. This is debug output left in shipped code: any model doing
adaptive integration spams a line per step (and a nontrivial step count
per call), and it can't be turned off short of editing the library. It has
no functional effect on the result — just stray, unconditional console
output on a hot numerical path.

---

## 10. `Tools/ODESolver/ODESolver.java` — dead duplicate `else if (dt < dtMin)` branch in the clamped `Runge45`

**Location:** `HAL/Tools/ODESolver/ODESolver.java`, in the fixed-output
`Runge45(..., double dtMin, double dtMax)` overload, inside the
`error > errorTolerance` retry block

```java
if (dt < dtMin) {
    dt = dtMin;
} else if (dt < dtMin) {   // identical condition — can never be reached
    dt = dtMin;
}
```

The `else if` tests the exact same condition as the leading `if`, so it is
unreachable dead code (a copy/paste artifact). Harmless to behavior, but a
clear mistake — presumably one of the two was meant to test `dt > dtMax`.

---

## 11. `Tools/ODESolver/ODESolver.java` — separate-output `Runge45` evaluates every step at the original state, not the running result

**Location:** `HAL/Tools/ODESolver/ODESolver.java`, the
`Runge45(Derivative, double[] state, double[] out, ...)` overloads (both
the tolerance-only and the `dtMin`/`dtMax` variants)

```java
while (t < tf) {
    ...
    error = Runge45internal(Derivative, state, t, dt, errorTolerance); // always `state`
    ...
    if (first) {
        for (int i = 0; i < state.length; i++) { out[i] = state[i] + s7[i]; }
        first = false;
    } else {
        for (int i = 0; i < state.length; i++) { out[i] = out[i] + s7[i]; } // accumulates into `out`
    }
    t += dt;
    ...
}
```

Each step computes the RKF45 increment from `state` — the *original* input
array, never updated inside the loop — but accumulates that increment into
`out`. After the first step the two arrays diverge, so from the second step
on the method integrates using a stale derivative evaluated at the initial
condition rather than at the current point on the trajectory, giving an
incorrect result.

This is masked in all of HAL's own usage because the convenience overloads
(`Runge45(Derivative, state, t0, tf, ...)`) call this method with
`out == state` (the two references alias the same array), so
`Runge45internal(state, ...)` happens to read the up-to-date state and the
accumulation writes back into it. The bug only manifests when a caller
passes genuinely distinct `state` and `out` arrays — exactly what the
separate-output signature invites. The intended behavior is to evaluate
each step at the current running state (i.e. `out` after the first step),
which also makes the aliased and non-aliased calls agree.

---

## 12. `Tools/PhylogenyTracker/Genome.java` — `RemoveGenomeFromTree` leaves `listFirst` dangling at a removed genome

**Location:** `HAL/Tools/PhylogenyTracker/Genome.java`, `RemoveGenomeFromTree`

```java
if (prev != null) { prev.next = next; }
if (next != null) { next.prev = prev; }
// nothing updates myTracker.listFirst
```

The tracker's global doubly-linked genome list is headed by
`listFirst` (the most recently created genome). `RemoveGenomeFromTree`
unlinks the genome from its neighbours' `next`/`prev` pointers but never
checks whether the genome being removed *is* `listFirst`. When leaf removal
prunes the current head (a genome created and then killed before any newer
one is created), `listFirst` is left pointing at a genome that has been
removed from the tree (and is otherwise eligible for GC). The
next `NewMutantGenome` then does `child.next = listFirst`, re-linking the
removed genome back into the live list, and iteration (which starts at the
progenitor and walks `prev`) can be corrupted. The fix is to advance
`listFirst = next` when the removed node is the head.

This is hard to trigger in practice — it needs a genome to die as a
childless leaf while it is still the newest genome — and leaf removal is
off in all of HAL's own examples (it's incompatible with clone recording),
so the path is rarely exercised. But it is a real pointer-bookkeeping bug.

---

## 13. `Tools/PhylogenyTracker/PhylogenyTrackerInternal.java` — string identity (`==`) used for content comparison

**Location:** `HAL/Tools/PhylogenyTracker/PhylogenyTrackerInternal.java`,
in `RecordClones` and `WriteOutPops`

```java
if (timepointLabel == "CloneID" || timepointLabel == "ParentID") { ... }   // RecordClones
if (header == "CloneID" || header == "ParentID") { ... }                    // WriteOutPops
```

These guard against reserved column names by comparing strings with `==`,
which in Java tests *reference* identity, not content. It happens to catch
string **literals** (which the JVM interns, so `"CloneID" == "CloneID"` is
true) but silently fails for any label/header built at runtime (read from a
file, concatenated, etc.) that equals `"CloneID"`/`"ParentID"` — the very
case the check exists to reject. `.equals()` is required for content
comparison.

---

## 14. `Tools/Lenia/LeniaNPlayer.java` — `FindKernelDuplicates` only compares adjacent kernels and can reference a never-computed convolution

**Location:** `HAL/Tools/Lenia/LeniaNPlayer.java`, `FindKernelDuplicates`
(and its use of `fftFieldRefs` in `Update`)

```java
for (int i = 0; i < nPlayers; i++) {
    ...
    for (int k = 0; k < i; k++) {
        ...
        if (identical) { fftFieldRefs[i][j] = k; }
        else           { fftFieldRefs[i][j] = i; }   // overwritten every k
    }
}
```

This routine is an optimization: within a source column `j`, if two kernels
`K_ij` and `K_kj` are identical, the convolution `K*A_j` is computed once
and shared (`fftFieldRefs[i][j] = k` means "reuse well k's result"). But
`fftFieldRefs[i][j]` is reassigned on **every** iteration of the inner `k`
loop, so only the **last** comparison (`k == i-1`) survives. Two
consequences:

1. **Missed deduplication** (benign): a kernel identical to a non-adjacent
   earlier one (`K_ij == K_0j` but `K_ij != K_(i-1)j`) is not deduplicated,
   so it is recomputed — wasteful but correct.

2. **Stale read** (a real correctness bug): with three or more identical
   kernels in a column (`K_0j == K_1j == K_2j`), `fftFieldRefs[2][j]`
   becomes `1` (the last comparison), but well 1 itself was deduplicated
   (`fftFieldRefs[1][j] == 0`) and so its convolution buffer
   `fftFields[1][j]` is **never computed** that step. `Update` then reads
   `fftFields[fftFieldRefs[2][j]][j] == fftFields[1][j]`, i.e. stale/garbage
   data, for channel 2's growth — a wrong result.

The intended behavior is to point each duplicate at the *first* identical
kernel whose buffer is actually computed. Alternatively, the deduplication
could simply be dropped and every `K_ij*A_j` always computed (correct, and
simpler; the real optimization opportunity — caching `fft(A_j)` across
target channels — is orthogonal).

---

## 15. `GridsAndAgents/PDEGrid2D.java` — `GradientY` wraps along X instead of Y

**Location:** `HAL/GridsAndAgents/PDEGrid2D.java`, both `GradientY(int, int)`
and `GradientY(int, int, double)`

```java
double down = PDEequations.DisplacedY2D(field, x, y-1, xDim, yDim, wrapX, ...);
double up   = PDEequations.DisplacedY2D(field, x, y+1, xDim, yDim, wrapX, ...);
```

`DisplacedY2D`'s sixth argument is the wrap flag for the *y* direction, but
`GradientY` passes `wrapX`. On a grid that wraps differently on the two axes
(`wrapX != wrapY`), the y-direction gradient at a y-boundary wraps (or
doesn't) according to the wrong axis — e.g. a y-periodic field gives the
wrong gradient at `y == 0` / `y == yDim-1` if the x-axis isn't also
periodic. (The 3D `GradientY` correctly uses `wrapY`; only 2D has this.)

---

## 16. `GridsAndAgents/PDEGrid3D.java` — `GradientZ` displaces Z through the Y-axis helper

**Location:** `HAL/GridsAndAgents/PDEGrid3D.java`, both `GradientZ(...)`
overloads

```java
double down = PDEequations.DisplacedY3D(field, x, y, z-1, xDim, yDim, zDim, wrapZ, ...);
double up   = PDEequations.DisplacedY3D(field, x, y, z+1, xDim, yDim, zDim, wrapZ, ...);
```

`GradientZ` varies the z coordinate but calls `DisplacedY3D`, which
bounds-checks the *y* dimension (`InDim(y, yDim)`), not z. Since y is
unchanged and in-bounds, the helper returns `field[x*yDim*zDim + y*zDim + (z±1)]`
directly — with no z-boundary handling. At a z-boundary (`z-1 == -1` or
`z+1 == zDim`) that index lands on a neighboring cell in the flattened array
(e.g. `(x, y-1, zDim-1)`), so the gradient is computed from the wrong cell
rather than a wrapped / zero-flux boundary. `DisplacedZ3D` exists and is
what should have been called. (`GradientX`/`GradientY` in 3D use the correct
helpers; only `GradientZ` is wrong.)

---

## 17. `Interfaces/Grid3D.java` — `BucketFillMask` never marks the seed cell in its output

**Location:** `HAL/Interfaces/Grid3D.java`, `BucketFillMask`

```java
scratch2.add(iStart);
scratch[iStart]=true;                 // seed marked visited...
while(scratch2.size()>0){
    int i=scratch2.remove(scratch2.size()-1);
    ...
    int ct=MapHood(nearbyHood,x,y,z,(i2,x2,y2,z2)->!scratch[i2]);
    for(int j=0;j<ct;j++){
        int i3=nearbyHood[j];
        scratch[i3]=true;
        if(IsMask.Eval(i3,x3,y3,z3)){
            out[i3]=true;             // ...but out[] is only ever set for NEIGHBORS
            scratch2.add(i3);
        }
    }
}
```

`out[iStart]` is never assigned. The flood fill only writes `out[i3]=true`
for cells reached as a *neighbor* of an already-popped cell, so the seed
cell — which is popped first, before any neighbor pass can reach it — is
left `false` in the result even when it satisfies the mask. For a masked
region of a single cell (the seed alone) the function reports an empty
region; more generally the returned mask is always missing its own seed.
Callers must OR the seed back in themselves. (The 2D interface has no
`BucketFillMask`, so this is 3D-only.)

---

## 18. `Tools/Internal/PDEequations.java` — per-cell-velocity `Advection3`'s z direction is broken three ways

**Location:** `HAL/Tools/Internal/PDEequations.java`,
`Advection3(double[] field, double[] deltas, double[] xVels, double[] yVels,
double[] zVels, ...)` (the spatially-varying-velocity overload; the
uniform-velocity `Advection3` is fine)

```java
if (z == zDim - 1 && !wrapZ) {
    zVelp = DisplacedZ3D(zVels, x, y, z+1, xDim, yDim, zDim, wrapX, ZvelBC);   // (a) wrapX, not wrapZ
}
double zVelm = DisplacedY3D(zVels, x, y, z, xDim, yDim, zDim, wrapZ, ZvelBC); // (b) Y helper, and z, not z-1
...
zFluxp = zVelp * DisplacedY3D(field, x, y, z+1, xDim, yDim, zDim, wrapZ, BC); // (c) Y helper for a z displacement
...
zFluxm = zVelm * DisplacedY3D(field, x, y, z-1, xDim, yDim, zDim, wrapZ, BC); // (c) again
```

Three distinct defects, all in the z direction (x and y are correct):

- **(a)** the z-plus edge velocity lookup passes `wrapX` where `wrapZ`
  belongs, so the z edge wraps or doesn't based on the *x* axis's wrap
  setting.
- **(b)** the z-minus face velocity reads `DisplacedY3D(zVels, x, y, z, ...)`
  — the coordinate is `z`, not `z - 1`, so the "neighbor" velocity is just
  the center cell's own `zVels[i]`; the z-minus face never sees the actual
  z-1 cell's velocity.
- **(c)** every z-displaced *field* lookup goes through `DisplacedY3D`,
  which bounds-checks the **y** coordinate (`InDim(y, yDim)`), not z. With
  y unchanged and in-bounds it returns
  `field[x*yDim*zDim + y*zDim + (z±1)]` unconditionally: at a z boundary
  that index lands on a neighboring row of the flattened array — or, for
  the very last cell (`x = xDim-1, y = yDim-1, z = zDim-1` with a negative
  `zVelp`), one element **past the end of the array**
  (`ArrayIndexOutOfBoundsException`). Same defect class as #15/#16 (an
  axis mixed up with the Y helper), one function over.

The fix for all three is to route every z lookup through `DisplacedZ3D`
with the correct coordinate and `wrapZ`, symmetric with the x and y
directions.

---

## Reporting status

Not yet reported upstream. Reporting is a decision for the repo owner
(Geoff) to make and act on — this doc exists to keep a running, cite-able
list in the meantime, not to imply these have been filed.
