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
        // GPU-backed cell-bucket neighbor search plus a linear-spring
        // force accumulation, mirroring AgentGrid2D::GetAgentsRadApprox's
        // bucket scan (see include/Hal/Grids/AgentGrid2D.h) and
        // SphericalAgent2D::SumForces' interactionRad overload (see
        // include/Hal/Agents/SphericalAgent2D.h). Non-linear force laws
        // remain deferred. The IComputeBackend abstraction over
        // this kernel is Hal::Gpu::GpuAgentForceBackend2D
        // (include/Hal/Gpu/GpuAgentForceBackend2D.h).
        //
        // Agent positions are uploaded once per Dispatch() call as a flat
        // (x, y) array; there is no persistent agent-pool identity on the
        // GPU side (unlike the CPU AgentGrid2D/InternalAgentPool) — the
        // caller is responsible for keeping its own CPU-side agent order
        // in sync with the uploaded position/radius arrays and reading
        // the returned velocity deltas back by that same index.
        //
        // Four passes per Dispatch(), matching the shader files in
        // shaders/: AgentCellCount.comp (count agents per cell),
        // AgentPrefixSum.comp (turn counts into offsets, single
        // invocation — see that shader's header comment for why this is
        // an acceptable non-parallel step), AgentScatter.comp (place each
        // agent's index into its cell's slot range), AgentForceAccum.comp
        // (the actual neighbor scan + force sum).
        //
        // Buffers are VMA-allocated: every SSBO is device-local; one
        // persistently mapped
        // staging buffer carries the per-step agent upload and the
        // velocity readback, partitioned so a repeated Dispatch after one
        // UploadAgents stays valid. UploadAgents only writes the staging
        // buffer — the staging->device copies, a cell-count clear, all
        // four passes, and the readback copy are recorded into ONE
        // command buffer per Dispatch and waited on with a fence (not
        // vkQueueWaitIdle). The upload/readback per step itself is
        // inherent to this kernel's contract: agents move on the CPU
        // between steps, so unlike diffusion (see
        // DiffusionKernel2D::RunSteps) there is no GPU-resident
        // multi-step form to batch.
        class AgentForceKernel2D
        {
        public:
            AgentForceKernel2D(ComputeContext &context,
                               int32_t         xDim,
                               int32_t         yDim,
                               int32_t         maxAgents);
            ~AgentForceKernel2D();

            AgentForceKernel2D(const AgentForceKernel2D &other) = delete;
            AgentForceKernel2D &operator=(const AgentForceKernel2D &other) = delete;

            // positions is a flat (x0, y0, x1, y1, ...) array, radii one
            // entry per agent; both must have numAgents-derived lengths
            // and numAgents must not exceed the maxAgents given at
            // construction.
            void UploadAgents(const std::vector<float> &positions,
                              const std::vector<float> &radii,
                              int32_t                   numAgents);

            void Dispatch(float  interactionRad,
                          float  springConstant,
                          bool   wrapX,
                          bool   wrapY);

            // Flat (vx0, vy0, vx1, vy1, ...) array, one pair per agent
            // uploaded in the most recent UploadAgents() call.
            std::vector<float> DownloadVelocityDeltas() const;

        private:
            // Handles for one compute pipeline (shader module, its
            // descriptor set layout/pool/set, and its pipeline
            // layout/pipeline) — a plain grouping of related Vulkan
            // objects, not a polymorphic abstraction; each of the four
            // passes below builds and owns one of these.
            struct PipelineObjects
            {
                VkShaderModule        shaderModule;
                VkDescriptorSetLayout descriptorSetLayout;
                VkPipelineLayout      pipelineLayout;
                VkPipeline            pipeline;
                VkDescriptorPool      descriptorPool;
                VkDescriptorSet       descriptorSet;
            };

            void CreateBuffers();
            void CreateCellCountPipeline();
            void CreatePrefixSumPipeline();
            void CreateScatterPipeline();
            void CreateForceAccumPipeline();
            void CreateCommandObjects();
            void DestroyPipelineObjects(PipelineObjects &objects);

            // Ends m_commandBuffer, submits it, waits on m_fence, and
            // resets both for the next recording.
            void SubmitAndWait();

            ComputeContext &m_context;
            int32_t m_xDim;
            int32_t m_yDim;
            int32_t m_numCells;
            int32_t m_maxAgents;
            int32_t m_numAgents;

            VkBuffer      m_positionBuffer;
            VmaAllocation m_positionAllocation;
            VkBuffer      m_radiusBuffer;
            VmaAllocation m_radiusAllocation;
            VkBuffer      m_cellCountBuffer;
            VmaAllocation m_cellCountAllocation;
            VkBuffer      m_cellOffsetBuffer;
            VmaAllocation m_cellOffsetAllocation;
            VkBuffer      m_cellCursorBuffer;
            VmaAllocation m_cellCursorAllocation;
            VkBuffer      m_bucketAgentBuffer;
            VmaAllocation m_bucketAgentAllocation;
            VkBuffer      m_velocityDeltaBuffer;
            VmaAllocation m_velocityDeltaAllocation;

            // Persistently mapped host staging, partitioned as
            // [positions (2*maxAgents floats) | radii (maxAgents) |
            // velocity readback (2*maxAgents)].
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            PipelineObjects m_cellCountPipeline;
            PipelineObjects m_prefixSumPipeline;
            PipelineObjects m_scatterPipeline;
            PipelineObjects m_forceAccumPipeline;

            VkCommandPool m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence m_fence;
        };
    }
}
