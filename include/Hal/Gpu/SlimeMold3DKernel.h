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
        // Physarum rule parameters and the batch's random seed (see
        // shaders/SlimeMold3DAgents.comp and SlimeMold3DField.comp).
        struct SlimeMold3DParams
        {
            float    stepSize;      // grid units moved per step
            float    sensorAngle;   // half-angle of the forward sensor cone (rad)
            float    sensorDist;    // how far ahead agents sense (grid units)
            float    turnRate;      // 0..1 blend toward the strongest sensor
            float    depositAmount; // chemical dropped per agent per step
            float    jitter;        // random-wander magnitude
            float    diffuse;       // field diffusion coefficient (< 1/6 for 3D stability)
            float    decay;         // field multiplicative decay per step (< 1)
            uint32_t seed;          // base RNG seed for this batch
        };

        // GPU 3D Physarum ("slime mold"): agents sense a chemical field, steer
        // up its gradient, move, and deposit; the field then diffuses and decays.
        // The field stays on the device across calls; agents are uploaded and
        // read back each call.
        //
        // Agents are 8 floats: [0..2] position in grid coordinates, [3] the field
        // value at the agent (written back), [4..6] unit heading, [7] unused.
        class SlimeMold3DKernel
        {
        public:
            SlimeMold3DKernel(ComputeContext &context,
                              int32_t         xDim,
                              int32_t         yDim,
                              int32_t         zDim,
                              int32_t         agentCapacity);
            ~SlimeMold3DKernel();

            SlimeMold3DKernel(const SlimeMold3DKernel &other) = delete;
            SlimeMold3DKernel &operator=(const SlimeMold3DKernel &other) = delete;

            // Replaces the resident field with `field` (xDim*yDim*zDim floats).
            void SeedField(const std::vector<float> &field);

            // Downloads the current resident field (xDim*yDim*zDim floats).
            void ReadField(std::vector<float> &outField);

            // Runs `steps` (agent pass, field pass) pairs in one submission.
            // `agents` holds 8 floats per agent; agentCount may be 0 to advance only
            // the field. Throws std::invalid_argument if agentCount exceeds the
            // capacity or agents is too short.
            void RunSteps(std::vector<float>      &agents,
                          int32_t                  agentCount,
                          int32_t                  steps,
                          const SlimeMold3DParams &params);

            int32_t AgentCapacity() const { return m_agentCapacity; }

        private:
            void Destroy();

            void CreateBuffers();
            void InitializeFieldBuffers();
            void CreateDescriptorObjects();
            void CreatePipelines();
            void CreateCommandObjects();
            void SubmitAndWait();

            VkShaderModule LoadShaderModule(const char *pShaderName);
            VkPipeline     CreateComputePipeline(VkShaderModule   module,
                                                 VkPipelineLayout layout);

            ComputeContext &m_context;
            int32_t         m_xDim;
            int32_t         m_yDim;
            int32_t         m_zDim;
            int32_t         m_length;
            int32_t         m_agentCapacity;
            int32_t         m_fieldParity; // 0 => buffer A current, 1 => B

            VkBuffer      m_agentBuffer;
            VmaAllocation m_agentAllocation;
            VkBuffer      m_fieldBufferA;
            VmaAllocation m_fieldAllocationA;
            VkBuffer      m_fieldBufferB;
            VmaAllocation m_fieldAllocationB;
            VkBuffer      m_depositBuffer;
            VmaAllocation m_depositAllocation;
            VkBuffer      m_stagingBuffer;
            VmaAllocation m_stagingAllocation;
            void         *m_pStagingMapped;

            VkDescriptorSetLayout m_descriptorSetLayout;
            VkDescriptorPool      m_descriptorPool;
            VkDescriptorSet       m_agentSetReadA; // agents, fieldA, deposit
            VkDescriptorSet       m_agentSetReadB; // agents, fieldB, deposit
            VkDescriptorSet       m_fieldSetAToB;  // src A, dst B, deposit
            VkDescriptorSet       m_fieldSetBToA;  // src B, dst A, deposit

            VkShaderModule   m_agentShaderModule;
            VkShaderModule   m_fieldShaderModule;
            VkPipelineLayout m_agentPipelineLayout;
            VkPipelineLayout m_fieldPipelineLayout;
            VkPipeline       m_agentPipeline;
            VkPipeline       m_fieldPipeline;

            VkCommandPool   m_commandPool;
            VkCommandBuffer m_commandBuffer;
            VkFence         m_fence;
        };
    }
}
