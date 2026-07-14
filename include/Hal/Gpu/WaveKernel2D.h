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
        // GPU-resident 2D wave equation via explicit leapfrog integration
        // (see shaders/Wave2D.comp) — the hyperbolic counterpart to the
        // parabolic diffusion kernels: it propagates and interferes rather
        // than smoothing. This is the solver behind the GpuWaveField
        // ripple-tank showcase (examples/GpuWaveField.cpp).
        //
        // The scheme needs two time levels: the current field and the
        // previous one. RunSteps takes both as host arrays, uploads them,
        // advances `steps` leapfrog updates entirely on the GPU (ping-pong
        // between two device buffers, one dispatch per step), and reads
        // both back so the caller can inject sources and continue. Because
        // sources are stamped on the host each frame, the field round-trips
        // per frame by contract — but a whole batch of sub-steps between
        // injections stays GPU-resident in one submission.
        //
        // speed2 = (c*dt/dx)^2 is CFL-stable for speed2 <= 0.5; `damping`
        // (<= 1) bleeds energy per step. Same VMA device-local ping-pong +
        // fence-synchronized architecture as GrayScottKernel2D.
        class WaveKernel2D
        {
        public:
            WaveKernel2D(ComputeContext &context,
                         int32_t         xDim,
                         int32_t         yDim);
            ~WaveKernel2D();

            WaveKernel2D(const WaveKernel2D &other) = delete;
            WaveKernel2D &operator=(const WaveKernel2D &other) = delete;

            // Advances the wave field `steps` leapfrog updates. On entry
            // `cur` is the current field and `prev` the previous; on return
            // they hold the field after `steps` steps and the step before
            // it, ready to be passed back in next frame. Both are length
            // xDim*yDim in HAL's x*yDim + y order.
            void RunSteps(std::vector<float> &cur,
                          std::vector<float> &prev,
                          int32_t             steps,
                          float               speed2,
                          float               damping,
                          bool                wrap);

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
            VkDescriptorSet m_descriptorSetAcurBprev;
            VkDescriptorSet m_descriptorSetBcurAprev;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
