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
        // Parameters for one FluidKernel3D::RunSteps call. dt is in voxels
        // per step (velocities are voxels/step); the dissipations are
        // per-step multipliers (1 = lossless); buoyancy lifts hot fluid
        // along +y proportional to (temperature - ambientTemperature).
        // The source terms describe an analytic spherical divergence
        // source (an outward blast wind) centered at (sourceX, sourceY,
        // sourceZ) in voxel coordinates — set sourceStrength to 0 to
        // disable it. pressureIterations is rounded up to even so the
        // Jacobi ping-pong always lands the result in the same buffer.
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

        // GPU-resident incompressible fluid solver — Stam's "Stable
        // Fluids" (1999), the standard graphics model for smoke and fire:
        // semi-Lagrangian advection (FluidAdvect3D.comp, unconditionally
        // stable), buoyancy from an advected temperature field, and a
        // pressure Poisson projection (FluidDivergence3D.comp +
        // FluidJacobi3D.comp iterations + FluidProject3D.comp) that keeps
        // the velocity field divergence-free — the projection is what
        // produces the characteristic rolling, billowing motion. The
        // domain is a closed box (clamped stencils, zero normal velocity
        // at the walls), not toroidal.
        //
        // Velocity and pressure live ONLY on the device and persist
        // across RunSteps calls; the caller sees just the scalar fields
        // (density, temperature) it seeds and renders. Same VMA
        // device-local buffer + persistently mapped staging + fence-sync
        // architecture as GrayScott3DKernel: a whole batch of steps is
        // recorded into one command buffer with one scalar upload before
        // and one readback after.
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

            // Advances the fluid `steps` full solver steps (advect +
            // buoyancy, divergence + source, pressure solve, project).
            // scalars is interleaved (density, temperature) per voxel,
            // length 2*xDim*yDim*zDim in (x*yDim + y)*zDim + z order; it
            // is uploaded once, advected with the flow on the device, and
            // read back once. The velocity field carries over from the
            // previous call.
            void RunSteps(std::vector<float> &scalars,
                          int32_t             steps,
                          const FluidParams  &params);

        private:
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
