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
        // The Vulkan objects for one of the kernel's compute pipelines.
        struct PipelineObjects
        {
            VkShaderModule        shaderModule;
            VkDescriptorSetLayout descriptorSetLayout;
            VkPipelineLayout      pipelineLayout;
            VkPipeline            pipeline;
            VkDescriptorPool      descriptorPool;
            VkDescriptorSet       descriptorSet;
        };

        // GPU cell-bucket neighbor search and linear-spring force sum, matching
        // SphericalAgent2D::SumForces with force = springConstant * touchDist.
        // Positions and radii are uploaded per call as flat arrays, and velocity
        // deltas come back in the same agent order. Four passes per Dispatch:
        // count agents per cell, prefix-sum the counts into offsets, scatter
        // agent indices into their cells, then scan neighbors and sum forces.
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

            // positions is a flat (x0, y0, x1, y1, ...) array, radii one entry per
            // agent. Throws std::invalid_argument unless numAgents is in
            // [0, maxAgents] and both arrays hold that many agents.
            void UploadAgents(const std::vector<float> &positions,
                              const std::vector<float> &radii,
                              int32_t                   numAgents);

            void Dispatch(float  interactionRad,
                          float  springConstant,
                          bool   wrapX,
                          bool   wrapY);

            // Flat (vx0, vy0, vx1, vy1, ...) array, one pair per agent from the
            // most recent UploadAgents().
            std::vector<float> DownloadVelocityDeltas() const;

        private:
            void Destroy();

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
