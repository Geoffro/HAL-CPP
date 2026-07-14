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
        // GPU-backed 2D explicit diffusion, mirroring
        // Hal::Internal::Diffusion2's no-boundary-condition path (see
        // shaders/Diffusion2D.comp) — one compute-shader invocation per
        // grid cell. This is Phase 4's first compute kernel; Dirichlet
        // boundaries and discontinuous rates remain deferred (see
        // docs/PHASED_PORT.md).
        //
        // Buffers are VMA-allocated (docs/PHASED_PORT.md's Phase 9 VMA
        // item): the field/delta SSBOs are device-local, with one
        // persistently mapped host staging buffer carrying uploads and
        // readbacks — no per-call vkMapMemory, and submissions wait on a
        // fence rather than vkQueueWaitIdle.
        //
        // Two ways to run it:
        //  - UploadField / ClearDeltas / Dispatch / DownloadDeltas — one
        //    step per call with a host round-trip, the shape
        //    IDiffusionBackend2D::Diffuse needs.
        //  - RunSteps — N whole diffusion steps (each a diffusion pass
        //    plus a DiffusionUpdate2D.comp commit pass) recorded into ONE
        //    command buffer and submitted once: a single upload before
        //    and readback after, no host round-trips between steps. This
        //    is the batched answer to "frames-in-flight pipelining" for
        //    time-stepped compute — successive steps depend on each
        //    other, so overlapping independent frames doesn't apply;
        //    keeping the chain GPU-resident is what removes the
        //    per-step overhead.
        class DiffusionKernel2D
        {
        public:
            DiffusionKernel2D(ComputeContext &context,
                              int32_t         xDim,
                              int32_t         yDim);
            ~DiffusionKernel2D();

            DiffusionKernel2D(const DiffusionKernel2D &other) = delete;
            DiffusionKernel2D &operator=(const DiffusionKernel2D &other) = delete;

            void UploadField(const std::vector<float> &field);
            void ClearDeltas();
            void Dispatch(float  diffRate,
                          bool   wrapX,
                          bool   wrapY);
            std::vector<float> DownloadDeltas() const;

            // Runs `steps` full diffusion steps (stencil + committed
            // update, like PDEGrid2D::Diffusion + Update per step)
            // entirely on the GPU: field is uploaded once, evolved in
            // place, and written back once. Pending deltas are cleared at
            // the start.
            void RunSteps(std::vector<float> &field,
                          int32_t             steps,
                          float               diffRate,
                          bool                wrapX,
                          bool                wrapY);

        private:
            void CreateBuffers();
            void CreateDescriptorObjects();
            void CreatePipelines();
            void CreateCommandObjects();

            // Ends m_commandBuffer, submits it, waits on m_fence, and
            // resets both for the next recording.
            void SubmitAndWait();

            ComputeContext &m_context;
            int32_t m_xDim;
            int32_t m_yDim;
            int32_t m_length;

            VkBuffer      m_fieldBuffer;
            VmaAllocation m_fieldAllocation;
            VkBuffer      m_deltaBuffer;
            VmaAllocation m_deltaAllocation;
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            VkShaderModule m_shaderModule;
            VkShaderModule m_updateShaderModule;
            VkDescriptorSetLayout m_descriptorSetLayout;
            VkPipelineLayout m_pipelineLayout;
            VkPipeline m_pipeline;
            VkPipeline m_updatePipeline;
            VkDescriptorPool m_descriptorPool;
            VkDescriptorSet m_descriptorSet;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
