// Build (from the repository root; add -ccbin <gcc> if the default compiler is too new): nvcc -O3 -std=c++17 -arch=sm_89 -Icuda/src review/Claude/d1_check/bench_cub.cu cuda/src/sph_kernels.cu -o bench_cub
// Drop-in alternative to k_cell_scan: device-wide exclusive scan (CUB) + the cursor
// re-arm (cellCur = cellStart) + start[ncell].  Checked against the single-block kernel.
#include "sph.h"
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
int main()
{
    const int sizes[] = { 21850, 199500, 800000, 3100000 };
    for (int ncell : sizes)
    {
        std::vector<int> cnt(ncell, 0);
        for (int c = 0; c < ncell / 8; ++c) cnt[c] = 6;
        int *dCnt, *dStart, *dCnt2, *dStart2;
        cudaMalloc(&dCnt, ncell * sizeof(int)); cudaMalloc(&dCnt2, ncell * sizeof(int));
        cudaMalloc(&dStart, (ncell + 1) * sizeof(int)); cudaMalloc(&dStart2, (ncell + 1) * sizeof(int));
        void* tmp = nullptr; size_t tb = 0;
        cub::DeviceScan::ExclusiveSum(tmp, tb, dCnt, dStart, ncell);
        cudaMalloc(&tmp, tb);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float tot = 0.f; const int R = 50;
        for (int r = 0; r < R; ++r)
        {
            cudaMemcpy(dCnt, cnt.data(), ncell * sizeof(int), cudaMemcpyHostToDevice);
            cudaEventRecord(e0);
            // exclusive scan of the counts, then re-arm the scatter cursor (cellCur =
            // cellStart).  The solver also needs start[ncell] = total: give cellCur one
            // trailing zero slot and scan ncell + 1 entries.
            cub::DeviceScan::ExclusiveSum(tmp, tb, dCnt, dStart, ncell);
            cudaMemcpyAsync(dCnt, dStart, ncell * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); tot += ms;
        }
        // reference: the solver's single-block kernel on the same input
        cudaMemcpy(dCnt2, cnt.data(), ncell * sizeof(int), cudaMemcpyHostToDevice);
        launch_cell_scan(dStart2, dCnt2, ncell, 0);
        std::vector<int> a(ncell), b(ncell);
        cudaMemcpy(a.data(), dStart, ncell * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), dStart2, ncell * sizeof(int), cudaMemcpyDeviceToHost);
        printf("ncell %8d: CUB scan + cursor copy %.4f ms  (matches k_cell_scan: %s)\n",
               ncell, tot / R, a == b ? "yes" : "NO");
        cudaFree(tmp); cudaFree(dCnt); cudaFree(dCnt2); cudaFree(dStart); cudaFree(dStart2);
    }
}
