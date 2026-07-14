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
        // GPU-resident volumetric Gray-Scott reaction-diffusion (see
        // shaders/GrayScott3D.comp): two coupled chemical fields U and V on
        // a toroidal 3D grid, whose Turing instability grows the sponge /
        // coral / gyroid morphologies the Gpu3DReactionDiffusion showcase
        // renders. The 3D analogue of GrayScottKernel2D — same VMA
        // device-local ping-pong buffers (one interleaved vec2 per voxel),
        // persistently mapped staging, fence sync, and GPU-resident
        // RunSteps (a whole batch of steps in one submission, one upload and
        // one readback, reaction and both diffusions on the device).
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

            // Runs `steps` full Gray-Scott steps entirely on the GPU. Both
            // arrays are length xDim*yDim*zDim in (x*yDim + y)*zDim + z
            // order. feed/kill select the morphology; diffU/diffV are the
            // two diffusion rates (typically diffU > diffV).
            void RunSteps(std::vector<float> &u,
                          std::vector<float> &v,
                          int32_t             steps,
                          float               feed,
                          float               kill,
                          float               diffU,
                          float               diffV);

        private:
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
