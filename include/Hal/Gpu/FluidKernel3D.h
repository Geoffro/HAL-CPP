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
        // Parameters for one FluidKernel3D::RunSteps call. Velocities are in
        // voxels per step; the dissipations are per-step multipliers
        // (1 = lossless); buoyancy lifts fluid along +y in proportion to
        // (temperature - ambientTemperature). The source is a spherical
        // outward blast centered at (sourceX, sourceY, sourceZ) in voxels;
        // sourceStrength 0 disables it. pressureIterations is rounded up to
        // an even number.
        struct FluidParams
        {
            float   dt;
            float   velocityDissipation;
            float   densityDissipation;
            float   temperatureDissipation;
            float   buoyancy;
            float   ambientTemperature;
            float   sourceX;
            float   sourceY;
            float   sourceZ;
            float   sourceRadius;
            float   sourceStrength;
            int32_t pressureIterations;
        };

        // GPU incompressible fluid in a closed box, by Stam's stable fluids
        // (1999): semi-Lagrangian advection with buoyancy
        // (FluidAdvect3D.comp), then a Jacobi pressure projection
        // (FluidDivergence3D/Jacobi3D/Project3D.comp). Velocity and pressure
        // stay on the device across RunSteps calls; only the density and
        // temperature are uploaded and read back.
        class FluidKernel3D
        {
        public:
            FluidKernel3D(ComputeContext &context,
                          int32_t         xDim,
                          int32_t         yDim,
                          int32_t         zDim);
            ~FluidKernel3D();

            FluidKernel3D(const FluidKernel3D &other) = delete;
            FluidKernel3D &operator=(const FluidKernel3D &other) = delete;

            // Advances `steps` solver steps in one submission. scalars holds
            // (density, temperature) pairs per voxel, indexed
            // (x*yDim + y)*zDim + z; throws std::invalid_argument unless it
            // holds 2*xDim*yDim*zDim values.
            void RunSteps(std::vector<float> &scalars,
                          int32_t             steps,
                          const FluidParams  &params);

        private:
            void Destroy();

            void CreateBuffers();
            void CreateDescriptorObjects();
            void CreatePipelines();
            void CreateCommandObjects();
            void ZeroInitBuffers();

            VkPipeline CreatePipeline(const char       *pShaderName,
                                      VkShaderModule   &outShaderModule,
                                      VkPipelineLayout  pipelineLayout);

            // Ends m_commandBuffer, submits it, waits on m_fence, and
            // resets both for the next recording.
            void SubmitAndWait();

            ComputeContext &m_context;
            int32_t m_xDim;
            int32_t m_yDim;
            int32_t m_zDim;
            int32_t m_length;

            // Which velocity/scalar buffer holds the current state (0 =
            // the A buffers, 1 = the B buffers); flips every step and
            // persists across RunSteps calls with the velocity field.
            int32_t m_parity;

            VkBuffer      m_velBufferA;
            VmaAllocation m_velAllocationA;
            VkBuffer      m_velBufferB;
            VmaAllocation m_velAllocationB;
            VkBuffer      m_scalBufferA;
            VmaAllocation m_scalAllocationA;
            VkBuffer      m_scalBufferB;
            VmaAllocation m_scalAllocationB;
            VkBuffer      m_pressureBufferA;
            VmaAllocation m_pressureAllocationA;
            VkBuffer      m_pressureBufferB;
            VmaAllocation m_pressureAllocationB;
            VkBuffer      m_divergenceBuffer;
            VmaAllocation m_divergenceAllocation;
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            VkShaderModule m_advectShaderModule;
            VkShaderModule m_divergenceShaderModule;
            VkShaderModule m_jacobiShaderModule;
            VkShaderModule m_projectShaderModule;

            VkDescriptorSetLayout m_advectSetLayout;
            VkDescriptorSetLayout m_divergenceSetLayout;
            VkDescriptorSetLayout m_jacobiSetLayout;
            VkDescriptorSetLayout m_projectSetLayout;

            VkPipelineLayout m_advectPipelineLayout;
            VkPipelineLayout m_divergencePipelineLayout;
            VkPipelineLayout m_jacobiPipelineLayout;
            VkPipelineLayout m_projectPipelineLayout;

            VkPipeline m_advectPipeline;
            VkPipeline m_divergencePipeline;
            VkPipeline m_jacobiPipeline;
            VkPipeline m_projectPipeline;

            VkDescriptorPool m_descriptorPool;

            // Two of each, indexed by the step parity (advect,
            // divergence, project) or the Jacobi iteration parity.
            VkDescriptorSet m_advectSets[2];
            VkDescriptorSet m_divergenceSets[2];
            VkDescriptorSet m_jacobiSets[2];
            VkDescriptorSet m_projectSets[2];

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
