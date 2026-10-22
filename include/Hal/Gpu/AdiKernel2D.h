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
        // GPU implicit (ADI) 2D diffusion matching Hal::Internal::Diffusion2DADI
        // with zero-flux or periodic boundaries. Each step is two sweeps (x then
        // y); each line's tridiagonal system is solved by one workgroup with
        // parallel cyclic reduction (shaders/AdiSweep2D.comp), and periodic axes
        // use the Sherman-Morrison decomposition. RunSteps runs all steps in one
        // submission. Grid dimensions are capped at kMaxLine, and the device must
        // support the shader's workgroup size and shared memory (IsSupported).
        class AdiKernel2D
        {
        public:
            AdiKernel2D(ComputeContext &context,
                        int32_t         xDim,
                        int32_t         yDim);
            ~AdiKernel2D();

            AdiKernel2D(const AdiKernel2D &other) = delete;
            AdiKernel2D &operator=(const AdiKernel2D &other) = delete;

            // Runs `steps` ADI steps on the GPU; field is uploaded once and read
            // back once. Unconditionally stable.
            void RunSteps(std::vector<float> &field,
                          int32_t             steps,
                          float               diffRate,
                          bool                wrapX,
                          bool                wrapY);

            static const int32_t kMaxLine = 1024;

            // True if the device supports the workgroup size and shared
            // memory the solve needs.
            static bool IsSupported(ComputeContext &context);

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
