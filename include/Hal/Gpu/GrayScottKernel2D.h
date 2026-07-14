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
        // GPU-resident Gray-Scott reaction-diffusion (see
        // shaders/GrayScott2D.comp): two coupled chemical fields U and V
        // on a toroidal grid, whose Turing instability grows the coral,
        // maze, spot and mitosis patterns the GpuReactionDiffusion
        // showcase animates (examples/GpuReactionDiffusion.cpp).
        //
        // Same GPU-resident architecture as DiffusionKernel2D /
        // AdiKernel2D: VMA device-local buffers (two, ping-ponged between
        // steps) plus one persistently mapped staging buffer, fence
        // synchronized. RunSteps records every step of a batch into ONE
        // submission with a single upload and a single readback — the
        // reaction and both diffusions stay entirely on the GPU, so unlike
        // a CPU/GPU-split reaction-diffusion (which would pay an upload and
        // readback per step) there are no host round-trips inside a batch.
        // This is exactly the shape that makes the GPU pay off: the whole
        // time-stepped loop lives on the device.
        //
        // The two chemicals share one interleaved vec2 buffer (x = U,
        // y = V). RunSteps takes and returns the fields as separate flat
        // U/V arrays for the caller's convenience.
        class GrayScottKernel2D
        {
        public:
            GrayScottKernel2D(ComputeContext &context,
                              int32_t         xDim,
                              int32_t         yDim);
            ~GrayScottKernel2D();

            GrayScottKernel2D(const GrayScottKernel2D &other) = delete;
            GrayScottKernel2D &operator=(const GrayScottKernel2D &other) = delete;

            // Runs `steps` full Gray-Scott steps entirely on the GPU: the
            // U/V fields are uploaded once, evolved in place, and written
            // back once. feed/kill select the pattern regime; diffU/diffV
            // are the two diffusion rates (typically diffU > diffV). Both
            // arrays are length xDim*yDim in HAL's x*yDim + y order.
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
