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
        // GPU 2D wave equation by explicit leapfrog (shaders/Wave2D.comp).
        // speed2 = (c*dt/dx)^2 is stable for speed2 <= 0.5; damping (<= 1)
        // removes energy each step.
        class WaveKernel2D
        {
        public:
            WaveKernel2D(ComputeContext &context,
                         int32_t         xDim,
                         int32_t         yDim);
            ~WaveKernel2D();

            WaveKernel2D(const WaveKernel2D &other) = delete;
            WaveKernel2D &operator=(const WaveKernel2D &other) = delete;

            // Advances `steps` leapfrog updates on the GPU. cur and prev hold the
            // current and previous fields (xDim*yDim values, indexed x*yDim + y) and
            // are updated in place.
            void RunSteps(std::vector<float> &cur,
                          std::vector<float> &prev,
                          int32_t             steps,
                          float               speed2,
                          float               damping,
                          bool                wrap);

        private:
            void Destroy();

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
            VkDescriptorSet m_descriptorSetAcurBprev;
            VkDescriptorSet m_descriptorSetBcurAprev;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
