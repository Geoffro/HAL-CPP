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
        // GPU Gray-Scott reaction-diffusion on a toroidal 3D grid
        // (shaders/GrayScott3D.comp); see GrayScottKernel2D.
        class GrayScott3DKernel
        {
        public:
            GrayScott3DKernel(ComputeContext &context,
                              int32_t         xDim,
                              int32_t         yDim,
                              int32_t         zDim);
            ~GrayScott3DKernel();

            GrayScott3DKernel(const GrayScott3DKernel &other) = delete;
            GrayScott3DKernel &operator=(const GrayScott3DKernel &other) = delete;

            // Runs `steps` Gray-Scott steps on the GPU. u and v hold xDim*yDim*zDim
            // values each, indexed (x*yDim + y)*zDim + z.
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
            int32_t m_zDim;
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
