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
        // GPU Gray-Scott reaction-diffusion on a toroidal 2D grid
        // (shaders/GrayScott2D.comp). U and V share one interleaved buffer on the
        // device; RunSteps runs a batch of steps in one submission.
        class GrayScottKernel2D
        {
        public:
            GrayScottKernel2D(ComputeContext &context,
                              int32_t         xDim,
                              int32_t         yDim);
            ~GrayScottKernel2D();

            GrayScottKernel2D(const GrayScottKernel2D &other) = delete;
            GrayScottKernel2D &operator=(const GrayScottKernel2D &other) = delete;

            // Runs `steps` Gray-Scott steps on the GPU. feed/kill select the pattern
            // regime; diffU/diffV are the diffusion rates. u and v hold xDim*yDim
            // values each, indexed x*yDim + y.
            void RunSteps(std::vector<float> &u,
                          std::vector<float> &v,
                          int32_t             steps,
                          float               feed,
                          float               kill,
                          float               diffU,
                          float               diffV);

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
            VkDescriptorSet m_descriptorSetAToB;
            VkDescriptorSet m_descriptorSetBToA;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
