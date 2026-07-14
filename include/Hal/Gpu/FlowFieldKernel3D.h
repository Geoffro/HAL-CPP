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
        // GPU advection of a fixed pool of particles through a 3D
        // Arnold-Beltrami-Childress (ABC) flow (see shaders/FlowField3D.comp)
        // — a divergence-free, chaotically mixing velocity field. The solver
        // behind the GpuFlowField3D showcase, which renders the tracers as a
        // luminous 3D web of depth-tested spheres.
        //
        // A single device-local position buffer (vec4 per particle: xyz =
        // position, w = speed) is advanced in place, one dispatch per
        // sub-step, several sub-steps chained GPU-resident in one submission.
        // The particles round-trip per frame only because the host renders
        // them; the advection itself never leaves the device between the
        // upload and readback of a batch.
        class FlowFieldKernel3D
        {
        public:
            FlowFieldKernel3D(ComputeContext &context,
                              int32_t         particleCount);
            ~FlowFieldKernel3D();

            FlowFieldKernel3D(const FlowFieldKernel3D &other) = delete;
            FlowFieldKernel3D &operator=(const FlowFieldKernel3D &other) = delete;

            // Advances every particle `steps` RK2 sub-steps through the ABC
            // flow with coefficients (a, b, c) and time step dt. `positions`
            // is 4*particleCount floats (x, y, z, speed per particle); the
            // speed component is overwritten with the post-step speed.
            void RunSteps(std::vector<float> &positions,
                          int32_t             steps,
                          float               a,
                          float               b,
                          float               c,
                          float               dt);

            int32_t ParticleCount() const { return m_particleCount; }

        private:
            void CreateBuffers();
            void CreateDescriptorObjects();
            void CreatePipeline();
            void CreateCommandObjects();
            void SubmitAndWait();

            ComputeContext &m_context;
            int32_t m_particleCount;

            VkBuffer      m_positionBuffer;
            VmaAllocation m_positionAllocation;
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            VkShaderModule m_shaderModule;
            VkDescriptorSetLayout m_descriptorSetLayout;
            VkPipelineLayout m_pipelineLayout;
            VkPipeline m_pipeline;
            VkDescriptorPool m_descriptorPool;
            VkDescriptorSet m_descriptorSet;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
