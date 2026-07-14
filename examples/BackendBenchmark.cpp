// CPU/GPU timing comparison on shared workloads (the correctness
// cross-check between backends lives in the test suite; this adds the
// numbers). Headless: needs a Vulkan compute device but no window/display.
//
// Four workloads:
//   - explicit diffusion, ONE step per call through IDiffusionBackend2D
//     (each GPU call pays a full upload + dispatch + fence-wait +
//     readback round-trip plus a double<->float conversion — the price
//     of the interface's per-step contract);
//   - explicit diffusion, N steps GPU-RESIDENT via
//     DiffusionKernel2D::RunSteps (one upload, N stencil+commit pass
//     pairs in a single submission, one readback) against the same N
//     steps on the CPU — the realistic "run a PDE model" shape, and the
//     honest measure of what the GPU adds;
//   - implicit ADI diffusion, N steps GPU-RESIDENT via
//     AdiKernel2D::RunSteps (PCR tridiagonal solves, one workgroup per
//     line) against the CPU's per-line Thomas-algorithm sweeps;
//   - agent neighborhood forces per step through IAgentForceBackend2D
//     (the per-step upload/readback is inherent here: agents move on the
//     CPU between steps).
//
// Mcell-steps/s and Magent-steps/s columns give absolute throughput so
// the numbers mean something beyond the ratio.

#include "Hal/Agents/CpuAgentForceBackend2D.h"
#include "Hal/Agents/IAgentForceBackend2D.h"
#include "Hal/Gpu/AdiKernel2D.h"
#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/DiffusionKernel2D.h"
#include "Hal/Gpu/GpuAgentForceBackend2D.h"
#include "Hal/Gpu/GpuDiffusionBackend2D.h"
#include "Hal/Log/Logger.h"
#include "Hal/Pde/CpuAdiDiffusionBackend2D.h"
#include "Hal/Pde/CpuDiffusionBackend2D.h"
#include "Hal/Pde/IDiffusionBackend2D.h"
#include "Hal/Rand.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        double NowSeconds()
        {
            std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(t.time_since_epoch()).count();
        }

        // Times reps calls of backend.Diffuse on an n x n sinusoidal
        // field; returns seconds per call and writes the result field sum
        // into outChecksum (so the work can't be optimized away and the
        // two backends can be sanity-compared).
        double TimeDiffusion(IDiffusionBackend2D &backend,
                             int32_t              n,
                             int32_t              reps,
                             double              &outChecksum)
        {
            std::vector<double> field(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    field[static_cast<size_t>(x * n + y)] = 0.5 + 0.5 * std::sin(2.0 * 3.14159265358979323846 * (x + y) / n);
                }
            }
            std::vector<double> deltas(field.size(), 0.0);

            // Warmup (first GPU dispatch includes one-time setup costs).
            backend.Diffuse(field, deltas, 0.2, n, n, true, true);
            deltas.assign(field.size(), 0.0);

            double start = NowSeconds();
            for (int32_t r = 0; r < reps; r++)
            {
                backend.Diffuse(field, deltas, 0.2, n, n, true, true);
            }
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (double d : deltas)
            {
                outChecksum += d;
            }
            return elapsed / reps;
        }

        // Times a `steps`-step diffusion RUN on the CPU: one
        // Diffuse + commit (field += deltas, deltas = 0) per step, in
        // double precision — the per-step work PDEGrid2D::Diffusion +
        // Update does. Returns seconds per step.
        double TimeCpuDiffusionRun(int32_t  n,
                                   int32_t  steps,
                                   double  &outChecksum)
        {
            std::vector<double> field(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    field[static_cast<size_t>(x * n + y)] = 0.5 + 0.5 * std::sin(2.0 * 3.14159265358979323846 * (x + y) / n);
                }
            }
            std::vector<double> deltas(field.size(), 0.0);
            Hal::CpuDiffusionBackend2D backend;

            double start = NowSeconds();
            for (int32_t step = 0; step < steps; step++)
            {
                backend.Diffuse(field, deltas, 0.2, n, n, true, true);
                for (size_t i = 0; i < field.size(); i++)
                {
                    field[i] += deltas[i];
                    deltas[i] = 0.0;
                }
            }
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (double v : field)
            {
                outChecksum += v;
            }
            return elapsed / steps;
        }

        // The same `steps`-step run GPU-resident: one upload, all steps
        // chained in a single submission, one readback
        // (DiffusionKernel2D::RunSteps). Returns seconds per step.
        double TimeGpuDiffusionRun(Hal::Gpu::ComputeContext &context,
                                   int32_t                   n,
                                   int32_t                   steps,
                                   double                   &outChecksum)
        {
            std::vector<float> field(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    field[static_cast<size_t>(x * n + y)] = static_cast<float>(0.5 + 0.5 * std::sin(2.0 * 3.14159265358979323846 * (x + y) / n));
                }
            }
            Hal::Gpu::DiffusionKernel2D kernel(context, n, n);

            // Warmup covers pipeline/allocation first-use costs.
            kernel.RunSteps(field, 1, 0.2f, true, true);

            double start = NowSeconds();
            kernel.RunSteps(field, steps, 0.2f, true, true);
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (float v : field)
            {
                outChecksum += static_cast<double>(v);
            }
            return elapsed / steps;
        }

        // The same run shapes for the IMPLICIT (ADI) solver: CPU
        // DiffuseADI + commit per step vs the GPU-resident
        // AdiKernel2D::RunSteps (PCR tridiagonal solves per line).
        double TimeCpuAdiRun(int32_t  n,
                             int32_t  steps,
                             double  &outChecksum)
        {
            std::vector<double> field(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    field[static_cast<size_t>(x * n + y)] = 0.5 + 0.5 * std::sin(2.0 * 3.14159265358979323846 * (x + y) / n);
                }
            }
            std::vector<double> deltas(field.size(), 0.0);
            Hal::CpuAdiDiffusionBackend2D backend;

            double start = NowSeconds();
            for (int32_t step = 0; step < steps; step++)
            {
                backend.DiffuseADI(field, deltas, 0.8, n, n, true, true);
                for (size_t i = 0; i < field.size(); i++)
                {
                    field[i] += deltas[i];
                    deltas[i] = 0.0;
                }
            }
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (double v : field)
            {
                outChecksum += v;
            }
            return elapsed / steps;
        }

        double TimeGpuAdiRun(Hal::Gpu::ComputeContext &context,
                             int32_t                   n,
                             int32_t                   steps,
                             double                   &outChecksum)
        {
            std::vector<float> field(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int32_t x = 0; x < n; x++)
            {
                for (int32_t y = 0; y < n; y++)
                {
                    field[static_cast<size_t>(x * n + y)] = static_cast<float>(0.5 + 0.5 * std::sin(2.0 * 3.14159265358979323846 * (x + y) / n));
                }
            }
            Hal::Gpu::AdiKernel2D kernel(context, n, n);
            kernel.RunSteps(field, 1, 0.8f, true, true);

            double start = NowSeconds();
            kernel.RunSteps(field, steps, 0.8f, true, true);
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (float v : field)
            {
                outChecksum += static_cast<double>(v);
            }
            return elapsed / steps;
        }

        // Times reps calls of backend.AccumulateForces on numAgents
        // agents scattered over an n x n wrapped domain.
        double TimeAgentForces(IAgentForceBackend2D &backend,
                               int32_t               n,
                               int32_t               numAgents,
                               int32_t               reps,
                               double               &outChecksum)
        {
            Rand rand(7);
            std::vector<double> positions;
            std::vector<double> radii;
            positions.reserve(static_cast<size_t>(numAgents) * 2);
            radii.reserve(static_cast<size_t>(numAgents));
            for (int32_t i = 0; i < numAgents; i++)
            {
                positions.push_back(rand.Double(static_cast<double>(n)));
                positions.push_back(rand.Double(static_cast<double>(n)));
                radii.push_back(0.5);
            }
            std::vector<double> deltas;

            backend.AccumulateForces(positions, radii, deltas, 2.0, 0.5, n, n, true, true);

            double start = NowSeconds();
            for (int32_t r = 0; r < reps; r++)
            {
                backend.AccumulateForces(positions, radii, deltas, 2.0, 0.5, n, n, true, true);
            }
            double elapsed = NowSeconds() - start;

            outChecksum = 0.0;
            for (double d : deltas)
            {
                outChecksum += std::fabs(d);
            }
            return elapsed / reps;
        }
    }
}

int main()
{
    try
    {
        std::printf("HAL C++ backend timing: CPU vs GPU on shared workloads\n\n");

        Hal::Gpu::ComputeContext context;

        std::printf("1) Explicit diffusion, ONE step per call (IDiffusionBackend2D):\n");
        std::printf("   every GPU call pays upload + dispatch + fence + readback.\n");
        std::printf("%10s %5s %14s %14s %9s\n", "grid", "reps", "CPU s/call", "GPU s/call", "speedup");
        const int32_t diffusionSizes[] = {64, 256, 512, 1024};
        for (int32_t n : diffusionSizes)
        {
            const int32_t reps = 20;
            Hal::CpuDiffusionBackend2D cpuBackend;
            Hal::Gpu::GpuDiffusionBackend2D gpuBackend(context, n, n);
            double cpuChecksum = 0.0;
            double gpuChecksum = 0.0;
            double cpuTime = Hal::Examples::TimeDiffusion(cpuBackend, n, reps, cpuChecksum);
            double gpuTime = Hal::Examples::TimeDiffusion(gpuBackend, n, reps, gpuChecksum);
            std::printf("%6dx%-4d %5d %14.6f %14.6f %8.2fx\n", n, n, reps, cpuTime, gpuTime, cpuTime / gpuTime);
        }

        std::printf("\n2) Explicit diffusion, %d-step RUN (the realistic PDE-model shape):\n", 100);
        std::printf("   CPU: Diffuse + commit per step. GPU: DiffusionKernel2D::RunSteps —\n");
        std::printf("   one upload, all steps in one submission, one readback.\n");
        std::printf("%10s %14s %14s %9s %12s %12s\n", "grid", "CPU s/step", "GPU s/step", "speedup", "CPU Mcell/s", "GPU Mcell/s");
        const int32_t runSizes[] = {256, 512, 1024, 2048};
        for (int32_t n : runSizes)
        {
            const int32_t steps = 100;
            double cpuChecksum = 0.0;
            double gpuChecksum = 0.0;
            double cpuTime = Hal::Examples::TimeCpuDiffusionRun(n, steps, cpuChecksum);
            double gpuTime = Hal::Examples::TimeGpuDiffusionRun(context, n, steps, gpuChecksum);
            double cells = static_cast<double>(n) * static_cast<double>(n);
            double drift = std::fabs(cpuChecksum - gpuChecksum) / (std::fabs(cpuChecksum) + 1e-12);
            std::printf("%6dx%-4d %14.6f %14.6f %8.2fx %12.1f %12.1f   (checksum drift %.2e)\n",
                        n, n, cpuTime, gpuTime, cpuTime / gpuTime, cells / cpuTime / 1e6, cells / gpuTime / 1e6, drift);
        }

        std::printf("\n3) Implicit ADI diffusion, %d-step RUN (rate 0.8 — no CFL cap):\n", 50);
        std::printf("   CPU: Thomas-algorithm sweeps per line. GPU: AdiKernel2D::RunSteps —\n");
        std::printf("   PCR tridiagonal solves, one workgroup per line, GPU-resident.\n");
        std::printf("%10s %14s %14s %9s %12s %12s\n", "grid", "CPU s/step", "GPU s/step", "speedup", "CPU Mcell/s", "GPU Mcell/s");
        const int32_t adiSizes[] = {256, 512, 1024};
        for (int32_t n : adiSizes)
        {
            const int32_t steps = 50;
            double cpuChecksum = 0.0;
            double gpuChecksum = 0.0;
            double cpuTime = Hal::Examples::TimeCpuAdiRun(n, steps, cpuChecksum);
            double gpuTime = Hal::Examples::TimeGpuAdiRun(context, n, steps, gpuChecksum);
            double cells = static_cast<double>(n) * static_cast<double>(n);
            double drift = std::fabs(cpuChecksum - gpuChecksum) / (std::fabs(cpuChecksum) + 1e-12);
            std::printf("%6dx%-4d %14.6f %14.6f %8.2fx %12.1f %12.1f   (checksum drift %.2e)\n",
                        n, n, cpuTime, gpuTime, cpuTime / gpuTime, cells / cpuTime / 1e6, cells / gpuTime / 1e6, drift);
        }

        std::printf("\n4) Agent neighborhood forces per step (IAgentForceBackend2D,\n");
        std::printf("   256x256 domain, rad 2.0; per-step upload/readback is inherent —\n");
        std::printf("   agents move on the CPU between steps):\n");
        std::printf("%10s %5s %14s %14s %9s %14s\n", "agents", "reps", "CPU s/call", "GPU s/call", "speedup", "GPU Magent/s");
        const int32_t agentCounts[] = {1000, 10000, 50000, 200000};
        for (int32_t numAgents : agentCounts)
        {
            const int32_t n = 256;
            const int32_t reps = 10;
            Hal::CpuAgentForceBackend2D cpuBackend;
            Hal::Gpu::GpuAgentForceBackend2D gpuBackend(context, n, n, numAgents);
            double cpuChecksum = 0.0;
            double gpuChecksum = 0.0;
            double cpuTime = Hal::Examples::TimeAgentForces(cpuBackend, n, numAgents, reps, cpuChecksum);
            double gpuTime = Hal::Examples::TimeAgentForces(gpuBackend, n, numAgents, reps, gpuChecksum);
            double checksumDrift = std::fabs(cpuChecksum - gpuChecksum) / ((cpuChecksum + gpuChecksum) * 0.5 + 1e-12);
            std::printf("%10d %5d %14.6f %14.6f %8.2fx %14.2f   (checksum drift %.2e)\n",
                        numAgents, reps, cpuTime, gpuTime, cpuTime / gpuTime, static_cast<double>(numAgents) / gpuTime / 1e6, checksumDrift);
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        Hal::Logger::Instance().Log(Hal::LogSeverity::Error, std::string("BackendBenchmark failed (no usable Vulkan compute device?): ") + e.what());
        return 1;
    }
}
