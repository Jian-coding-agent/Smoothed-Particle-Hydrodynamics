// Build (from the repository root; add -ccbin <gcc> if the default compiler is too new): nvcc -O3 -std=c++17 -arch=sm_89 -Icuda/src review/Claude/d1_check/bench_grid.cu cuda/src/sph_kernels.cu -o bench_grid
// Per-step cost of the cell-list bookkeeping that scales with the number of grid
// cells (not particles): 3 memsets + k_cell_scan (one 512-thread block) +
// k_cell_sort (one thread per cell).  V2 gives every rank the *global* grid, so a
// rank pays this for every cell of the domain even though it occupies ~1/K of them.
// Links against the solver's own sph_kernels.cu.
#include "sph.h"
#include <vector>
#include <cstdio>
#include <chrono>

int main()
{
    const int sizes[] = { 21850, 199500, 800000, 3100000 };
    for (int ncell : sizes)
    {
        // a rank's particles fill the first 1/8 of the cells, ~6 per cell
        const int occupied = ncell / 8, per = 6, np = occupied * per;
        std::vector<int> cnt(ncell, 0), start(ncell + 1, 0), sorted(np), gid(np);
        for (int c = 0; c < occupied; ++c) cnt[c] = per;
        int run = 0;
        for (int c = 0; c < ncell; ++c) { start[c] = run; run += cnt[c]; }
        start[ncell] = run;
        for (int i = 0; i < np; ++i) { sorted[i] = np - 1 - i; gid[i] = i; }   // reverse order: worst case for insertion sort

        int *dCnt, *dStart, *dSorted, *dGid, *dA, *dB, *dC;
        cudaMalloc(&dCnt, ncell * sizeof(int)); cudaMalloc(&dStart, (ncell + 1) * sizeof(int));
        cudaMalloc(&dSorted, np * sizeof(int)); cudaMalloc(&dGid, np * sizeof(int));
        cudaMalloc(&dA, ncell * sizeof(int)); cudaMalloc(&dB, ncell * sizeof(int)); cudaMalloc(&dC, ncell * sizeof(int));
        cudaMemcpy(dGid, gid.data(), np * sizeof(int), cudaMemcpyHostToDevice);

        auto once = [&]()
        {
            cudaMemsetAsync(dA, 0, ncell * sizeof(int), 0);
            cudaMemsetAsync(dB, 0, ncell * sizeof(int), 0);
            cudaMemsetAsync(dC, 0, ncell * sizeof(int), 0);
            cudaMemcpyAsync(dCnt, cnt.data(), ncell * sizeof(int), cudaMemcpyHostToDevice, 0);
            cudaMemcpyAsync(dSorted, sorted.data(), np * sizeof(int), cudaMemcpyHostToDevice, 0);
        };
        auto timed = [&](auto&& fn, int R)
        {
            cudaDeviceSynchronize();
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            float tot = 0.f;
            for (int r = 0; r < R; ++r)
            {
                once(); cudaDeviceSynchronize();
                cudaEventRecord(e0, 0); fn(); cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
                float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
            }
            return tot / R;
        };
        const float tMem = timed([&] { cudaMemsetAsync(dA, 0, ncell * sizeof(int), 0);
                                       cudaMemsetAsync(dB, 0, ncell * sizeof(int), 0);
                                       cudaMemsetAsync(dC, 0, ncell * sizeof(int), 0); }, 50);
        const float tScan = timed([&] { launch_cell_scan(dStart, dCnt, ncell, 0); }, 50);
        // sort needs the real start[] array
        cudaMemcpy(dStart, start.data(), (ncell + 1) * sizeof(int), cudaMemcpyHostToDevice);
        const float tSort = timed([&] { launch_cell_sort(dStart, dSorted, ncell, dGid, 0); }, 50);
        printf("ncell %8d (rank holds %7d particles): memsets %.3f ms, scan %.3f ms, sort %.3f ms, total %.3f ms\n",
               ncell, np, tMem, tScan, tSort, tMem + tScan + tSort);
        cudaFree(dCnt); cudaFree(dStart); cudaFree(dSorted); cudaFree(dGid); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
