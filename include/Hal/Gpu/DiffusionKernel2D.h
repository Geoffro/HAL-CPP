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
        // GPU explicit 2D diffusion matching Hal::Internal::Diffusion2's
        // zero-flux/periodic form (shaders/Diffusion2D.comp). Two ways to run it:
        //  - UploadField / ClearDeltas / Dispatch / DownloadDeltas: one step per
        //    call with a host round trip (what IDiffusionBackend2D needs).
        //  - RunSteps: many full steps (stencil plus commit) in one submission
        //    with a single upload and readback, which is far faster.
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

            // Runs `steps` full diffusion steps (like PDEGrid2D::Diffusion +
            // Update) on the GPU; field is uploaded once and read back once.
            // Pending deltas are cleared first.
            void RunSteps(std::vector<float> &field,
                          int32_t             steps,
                          float               diffRate,
                          bool                wrapX,
                          bool                wrapY);

        private:
            void Destroy();

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
