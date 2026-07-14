#pragma once

#include "Hal/Gpu/ComputeContext.h"
#include "Hal/Gpu/Internal/VulkanUtil.h"

#include <volk.h>

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        // GPU implicit (ADI / Crank-Nicolson) 2D diffusion — the
        // tridiagonal-solve-on-GPU stretch goal, mirroring
        // Hal::Internal::Diffusion2DADI: each step is two
        // directional sweeps (solve along x per row with the explicit y
        // half-step as RHS, then along y per column), each line's
        // tridiagonal system solved by one workgroup with PCR (parallel
        // cyclic reduction) in shared memory; wrapped axes go through the
        // same Sherman-Morrison decomposition TdmaSolver::TDMAperiodic
        // uses (see shaders/AdiSweep2D.comp).
        //
        // Same architecture as the reworked DiffusionKernel2D: VMA
        // device-local buffers (two, ping-ponged between sweeps) plus one
        // persistently mapped staging buffer, fence-synchronized, and
        // GPU-resident stepping — RunSteps records all sweeps of all
        // steps into ONE submission with one upload and one readback.
        //
        // Because each line lives in one workgroup's shared memory, grid
        // dimensions are capped at kMaxLine (1024, matching the shader's
        // fixed workgroup size); the constructor validates the dimensions
        // and the device's workgroup limits and throws if unsupported.
        class AdiKernel2D
        {
        public:
            AdiKernel2D(ComputeContext &context,
                        int32_t         xDim,
                        int32_t         yDim);
            ~AdiKernel2D();

            AdiKernel2D(const AdiKernel2D &other) = delete;
            AdiKernel2D &operator=(const AdiKernel2D &other) = delete;

            // Runs `steps` full ADI steps (two sweeps each) entirely on
            // the GPU: field is uploaded once, evolved in place, and
            // written back once. No CFL cap — ADI is unconditionally
            // stable, exactly like the CPU DiffusionADI.
            void RunSteps(std::vector<float> &field,
                          int32_t             steps,
                          float               diffRate,
                          bool                wrapX,
                          bool                wrapY);

            static const int32_t kMaxLine = 1024;

        private:
            void CreateBuffers();
            void CreateDescriptorObjects();
            void CreatePipeline();
            void CreateCommandObjects();
            void SubmitAndWait();

            ComputeContext &m_context;
            int32_t m_xDim;
            int32_t m_yDim;
            int32_t m_length;

            VkBuffer      m_bufferA;
            VmaAllocation m_allocationA;
            VkBuffer      m_bufferB;
            VmaAllocation m_allocationB;
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            VkShaderModule m_shaderModule;
            VkDescriptorSetLayout m_descriptorSetLayout;
            VkPipelineLayout m_pipelineLayout;
            VkPipeline m_pipeline;
            VkDescriptorPool m_descriptorPool;
            VkDescriptorSet m_descriptorSetAToB;
            VkDescriptorSet m_descriptorSetBToA;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
