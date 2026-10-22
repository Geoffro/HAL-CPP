#include "Hal/Gpu/SlimeMold3DKernel.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct SlimeAgentPushConstants
        {
            float    stepSize;
            float    sensorAngle;
            float    sensorDist;
            float    turnRate;
            float    depositAmount;
            float    jitter;
            int32_t  xDim;
            int32_t  yDim;
            int32_t  zDim;
            int32_t  agentCount;
            uint32_t seed;
        };

        struct SlimeFieldPushConstants
        {
            float   diffuse;
            float   decay;
            int32_t xDim;
            int32_t yDim;
            int32_t zDim;
        };

        // The agent pass runs a linear workgroup of 256 over the agents; the
        // field pass runs an 8x8x8 workgroup over the voxels.
        const uint32_t g_slimeAgentGroupSize = 256;
        const uint32_t g_slimeFieldGroupSize = 8;

        SlimeMold3DKernel::SlimeMold3DKernel(ComputeContext &context,
                                             int32_t         xDim,
                                             int32_t         yDim,
                                             int32_t         zDim,
                                             int32_t         agentCapacity)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_zDim(zDim),
              m_length(xDim * yDim * zDim),
              m_agentCapacity(agentCapacity),
              m_fieldParity(0),
              m_agentBuffer(VK_NULL_HANDLE),
              m_agentAllocation(VK_NULL_HANDLE),
              m_fieldBufferA(VK_NULL_HANDLE),
              m_fieldAllocationA(VK_NULL_HANDLE),
              m_fieldBufferB(VK_NULL_HANDLE),
              m_fieldAllocationB(VK_NULL_HANDLE),
              m_depositBuffer(VK_NULL_HANDLE),
              m_depositAllocation(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_descriptorSetLayout(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_agentSetReadA(VK_NULL_HANDLE),
              m_agentSetReadB(VK_NULL_HANDLE),
              m_fieldSetAToB(VK_NULL_HANDLE),
              m_fieldSetBToA(VK_NULL_HANDLE),
              m_agentShaderModule(VK_NULL_HANDLE),
              m_fieldShaderModule(VK_NULL_HANDLE),
              m_agentPipelineLayout(VK_NULL_HANDLE),
              m_fieldPipelineLayout(VK_NULL_HANDLE),
              m_agentPipeline(VK_NULL_HANDLE),
              m_fieldPipeline(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_commandBuffer(VK_NULL_HANDLE),
              m_fence(VK_NULL_HANDLE)
        {
            try
            {
                CreateBuffers();
                CreateDescriptorObjects();
                CreatePipelines();
                CreateCommandObjects();
                InitializeFieldBuffers();
            }
            catch (...)
            {
                Destroy();
                throw;
            }
        }

        SlimeMold3DKernel::~SlimeMold3DKernel()
        {
            Destroy();
        }

        void SlimeMold3DKernel::Destroy()
        {
            VkDevice device = m_context.Device();
            if (m_fence != VK_NULL_HANDLE)
            {
                vkDestroyFence(device, m_fence, nullptr);
            }
            if (m_commandPool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(device, m_commandPool, nullptr);
            }
            if (m_agentPipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_agentPipeline, nullptr);
            }
            if (m_fieldPipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_fieldPipeline, nullptr);
            }
            if (m_agentPipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_agentPipelineLayout, nullptr);
            }
            if (m_fieldPipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_fieldPipelineLayout, nullptr);
            }
            if (m_agentShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_agentShaderModule, nullptr);
            }
            if (m_fieldShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_fieldShaderModule, nullptr);
            }
            if (m_descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
            }
            if (m_descriptorSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
            }
            Internal::DestroyBufferVma(m_context.Allocator(), m_stagingBuffer, m_stagingAllocation);
            Internal::DestroyBufferVma(m_context.Allocator(), m_depositBuffer, m_depositAllocation);
            Internal::DestroyBufferVma(m_context.Allocator(), m_fieldBufferB, m_fieldAllocationB);
            Internal::DestroyBufferVma(m_context.Allocator(), m_fieldBufferA, m_fieldAllocationA);
            Internal::DestroyBufferVma(m_context.Allocator(), m_agentBuffer, m_agentAllocation);
        }

        void SlimeMold3DKernel::CreateBuffers()
        {
            VkDeviceSize agentSize = static_cast<VkDeviceSize>(m_agentCapacity) * 8 * sizeof(float);
            VkDeviceSize fieldSize = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            VkDeviceSize depositSize = static_cast<VkDeviceSize>(m_length) * sizeof(uint32_t);
            VkDeviceSize stagingSize = (agentSize > fieldSize) ? agentSize : fieldSize;

            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_agentBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), agentSize, deviceUsage, m_agentAllocation);
            m_fieldBufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), fieldSize, deviceUsage, m_fieldAllocationA);
            m_fieldBufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), fieldSize, deviceUsage, m_fieldAllocationB);
            m_depositBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), depositSize, deviceUsage, m_depositAllocation);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void SlimeMold3DKernel::CreateDescriptorObjects()
        {
            VkDevice device = m_context.Device();

            // Both passes use the same three-storage-buffer layout; only the
            // buffers bound (and the push-constant block) differ.
            VkDescriptorSetLayoutBinding bindings[3] = {};
            for (int32_t b = 0; b < 3; b++)
            {
                bindings[b].binding = static_cast<uint32_t>(b);
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 3;
            layoutInfo.pBindings = bindings;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 12; // 4 sets * 3 bindings

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 4;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetLayout setLayouts[4] = {m_descriptorSetLayout, m_descriptorSetLayout, m_descriptorSetLayout, m_descriptorSetLayout};
            VkDescriptorSet sets[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = m_descriptorPool;
            setAllocInfo.descriptorSetCount = 4;
            setAllocInfo.pSetLayouts = setLayouts;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, sets), "vkAllocateDescriptorSets failed");
            m_agentSetReadA = sets[0];
            m_agentSetReadB = sets[1];
            m_fieldSetAToB = sets[2];
            m_fieldSetBToA = sets[3];

            VkDescriptorBufferInfo agentInfo = {};
            agentInfo.buffer = m_agentBuffer;
            agentInfo.offset = 0;
            agentInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo fieldAInfo = {};
            fieldAInfo.buffer = m_fieldBufferA;
            fieldAInfo.offset = 0;
            fieldAInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo fieldBInfo = {};
            fieldBInfo.buffer = m_fieldBufferB;
            fieldBInfo.offset = 0;
            fieldBInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo depositInfo = {};
            depositInfo.buffer = m_depositBuffer;
            depositInfo.offset = 0;
            depositInfo.range = VK_WHOLE_SIZE;

            // (set, binding, buffer) triples for all four sets.
            // Agent sets: b0 = agents, b1 = field src, b2 = deposit.
            // Field sets: b0 = field src, b1 = field dst, b2 = deposit.
            struct Assign
            {
                VkDescriptorSet         set;
                uint32_t                binding;
                VkDescriptorBufferInfo *pInfo;
            };
            Assign assigns[12] = {
                {m_agentSetReadA, 0, &agentInfo},
                {m_agentSetReadA, 1, &fieldAInfo},
                {m_agentSetReadA, 2, &depositInfo},
                {m_agentSetReadB, 0, &agentInfo},
                {m_agentSetReadB, 1, &fieldBInfo},
                {m_agentSetReadB, 2, &depositInfo},
                {m_fieldSetAToB, 0, &fieldAInfo},
                {m_fieldSetAToB, 1, &fieldBInfo},
                {m_fieldSetAToB, 2, &depositInfo},
                {m_fieldSetBToA, 0, &fieldBInfo},
                {m_fieldSetBToA, 1, &fieldAInfo},
                {m_fieldSetBToA, 2, &depositInfo},
            };

            VkWriteDescriptorSet writes[12] = {};
            for (int32_t w = 0; w < 12; w++)
            {
                writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[w].dstSet = assigns[w].set;
                writes[w].dstBinding = assigns[w].binding;
                writes[w].descriptorCount = 1;
                writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[w].pBufferInfo = assigns[w].pInfo;
            }
            vkUpdateDescriptorSets(device, 12, writes, 0, nullptr);
        }

        VkShaderModule SlimeMold3DKernel::LoadShaderModule(const char *pShaderName)
        {
            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath(pShaderName));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            VkShaderModule module = VK_NULL_HANDLE;
            Internal::CheckResult(vkCreateShaderModule(m_context.Device(), &shaderInfo, nullptr, &module), "vkCreateShaderModule failed");
            return module;
        }

        VkPipeline SlimeMold3DKernel::CreateComputePipeline(VkShaderModule   module,
                                                            VkPipelineLayout layout)
        {
            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = module;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = layout;
            VkPipeline pipeline = VK_NULL_HANDLE;
            Internal::CheckResult(vkCreateComputePipelines(m_context.Device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines failed");
            return pipeline;
        }

        void SlimeMold3DKernel::CreatePipelines()
        {
            VkDevice device = m_context.Device();

            m_agentShaderModule = LoadShaderModule("SlimeMold3DAgents.comp");
            m_fieldShaderModule = LoadShaderModule("SlimeMold3DField.comp");

            VkPushConstantRange agentRange = {};
            agentRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            agentRange.offset = 0;
            agentRange.size = sizeof(SlimeAgentPushConstants);
            VkPipelineLayoutCreateInfo agentLayoutInfo = {};
            agentLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            agentLayoutInfo.setLayoutCount = 1;
            agentLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
            agentLayoutInfo.pushConstantRangeCount = 1;
            agentLayoutInfo.pPushConstantRanges = &agentRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &agentLayoutInfo, nullptr, &m_agentPipelineLayout), "vkCreatePipelineLayout failed");

            VkPushConstantRange fieldRange = {};
            fieldRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            fieldRange.offset = 0;
            fieldRange.size = sizeof(SlimeFieldPushConstants);
            VkPipelineLayoutCreateInfo fieldLayoutInfo = {};
            fieldLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            fieldLayoutInfo.setLayoutCount = 1;
            fieldLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
            fieldLayoutInfo.pushConstantRangeCount = 1;
            fieldLayoutInfo.pPushConstantRanges = &fieldRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &fieldLayoutInfo, nullptr, &m_fieldPipelineLayout), "vkCreatePipelineLayout failed");

            m_agentPipeline = CreateComputePipeline(m_agentShaderModule, m_agentPipelineLayout);
            m_fieldPipeline = CreateComputePipeline(m_fieldShaderModule, m_fieldPipelineLayout);
        }

        void SlimeMold3DKernel::CreateCommandObjects()
        {
            VkDevice device = m_context.Device();

            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = m_context.ComputeQueueFamilyIndex();
            Internal::CheckResult(vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool), "vkCreateCommandPool failed");

            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            Internal::CheckResult(vkAllocateCommandBuffers(device, &allocInfo, &m_commandBuffer), "vkAllocateCommandBuffers failed");

            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            Internal::CheckResult(vkCreateFence(device, &fenceInfo, nullptr, &m_fence), "vkCreateFence failed");
        }

        void SlimeMold3DKernel::InitializeFieldBuffers()
        {
            // The field and deposit buffers are device-local; clear them to
            // zero once up front so the first agent pass senses an empty field
            // and the deposit accumulator starts balanced.
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            vkCmdFillBuffer(m_commandBuffer, m_fieldBufferA, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_fieldBufferB, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_depositBuffer, 0, VK_WHOLE_SIZE, 0);

            SubmitAndWait();
            m_fieldParity = 0;
        }

        void SlimeMold3DKernel::SubmitAndWait()
        {
            Internal::CheckResult(vkEndCommandBuffer(m_commandBuffer), "vkEndCommandBuffer failed");

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffer;
            Internal::CheckResult(vkQueueSubmit(m_context.ComputeQueue(), 1, &submitInfo, m_fence), "vkQueueSubmit failed");
            Internal::CheckResult(vkWaitForFences(m_context.Device(), 1, &m_fence, VK_TRUE, std::numeric_limits<uint64_t>::max()), "vkWaitForFences failed");
            Internal::CheckResult(vkResetFences(m_context.Device(), 1, &m_fence), "vkResetFences failed");
            Internal::CheckResult(vkResetCommandBuffer(m_commandBuffer, 0), "vkResetCommandBuffer failed");
        }

        void SlimeMold3DKernel::SeedField(const std::vector<float> &field)
        {
            Internal::CheckSize(field.size(), static_cast<size_t>(m_length), "SlimeMold3DKernel::SeedField field");
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            std::memcpy(m_pStagingMapped, field.data(), static_cast<size_t>(size));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBuffer target = (m_fieldParity == 0) ? m_fieldBufferA : m_fieldBufferB;
            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, target, 1, &region);

            SubmitAndWait();
        }

        void SlimeMold3DKernel::ReadField(std::vector<float> &outField)
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            outField.resize(static_cast<size_t>(m_length));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBuffer source = (m_fieldParity == 0) ? m_fieldBufferA : m_fieldBufferB;
            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, source, m_stagingBuffer, 1, &region);

            SubmitAndWait();
            std::memcpy(outField.data(), m_pStagingMapped, static_cast<size_t>(size));
        }

        void SlimeMold3DKernel::RunSteps(std::vector<float>      &agents,
                                         int32_t                  agentCount,
                                         int32_t                  steps,
                                         const SlimeMold3DParams &params)
        {
            Internal::CheckArgument((agentCount >= 0) && (agentCount <= m_agentCapacity), "SlimeMold3DKernel::RunSteps agentCount must be within [0, capacity]");
            Internal::CheckArgument(agents.size() >= (static_cast<size_t>(agentCount) * 8), "SlimeMold3DKernel::RunSteps agents must hold 8 floats per agent");
            VkDeviceSize agentSize = static_cast<VkDeviceSize>(agentCount) * 8 * sizeof(float);
            if (agentCount > 0)
            {
                std::memcpy(m_pStagingMapped, agents.data(), static_cast<size_t>(agentSize));
            }

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            if (agentCount > 0)
            {
                VkBufferCopy region = {};
                region.size = agentSize;
                vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_agentBuffer, 1, &region);

                VkMemoryBarrier uploadBarrier = {};
                uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploadBarrier, 0, nullptr, 0, nullptr);
            }

            VkMemoryBarrier computeBarrier = {};
            computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            SlimeAgentPushConstants agentPush = {};
            agentPush.stepSize = params.stepSize;
            agentPush.sensorAngle = params.sensorAngle;
            agentPush.sensorDist = params.sensorDist;
            agentPush.turnRate = params.turnRate;
            agentPush.depositAmount = params.depositAmount;
            agentPush.jitter = params.jitter;
            agentPush.xDim = m_xDim;
            agentPush.yDim = m_yDim;
            agentPush.zDim = m_zDim;
            agentPush.agentCount = agentCount;

            SlimeFieldPushConstants fieldPush = {};
            fieldPush.diffuse = params.diffuse;
            fieldPush.decay = params.decay;
            fieldPush.xDim = m_xDim;
            fieldPush.yDim = m_yDim;
            fieldPush.zDim = m_zDim;

            uint32_t agentGroups = (static_cast<uint32_t>(agentCount) + g_slimeAgentGroupSize - 1) / g_slimeAgentGroupSize;
            uint32_t fieldGroupsX = (static_cast<uint32_t>(m_xDim) + g_slimeFieldGroupSize - 1) / g_slimeFieldGroupSize;
            uint32_t fieldGroupsY = (static_cast<uint32_t>(m_yDim) + g_slimeFieldGroupSize - 1) / g_slimeFieldGroupSize;
            uint32_t fieldGroupsZ = (static_cast<uint32_t>(m_zDim) + g_slimeFieldGroupSize - 1) / g_slimeFieldGroupSize;

            for (int32_t step = 0; step < steps; step++)
            {
                // Agent pass: sense the current field, steer, move, deposit.
                if (agentGroups > 0)
                {
                    VkDescriptorSet agentSet = (m_fieldParity == 0) ? m_agentSetReadA : m_agentSetReadB;
                    agentPush.seed = params.seed + static_cast<uint32_t>(step);
                    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_agentPipeline);
                    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_agentPipelineLayout, 0, 1, &agentSet, 0, nullptr);
                    vkCmdPushConstants(m_commandBuffer, m_agentPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SlimeAgentPushConstants), &agentPush);
                    vkCmdDispatch(m_commandBuffer, agentGroups, 1, 1);
                    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
                }

                // Field pass: diffuse + decay the current field into the other
                // buffer, folding in and clearing this step's deposit.
                VkDescriptorSet fieldSet = (m_fieldParity == 0) ? m_fieldSetAToB : m_fieldSetBToA;
                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_fieldPipeline);
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_fieldPipelineLayout, 0, 1, &fieldSet, 0, nullptr);
                vkCmdPushConstants(m_commandBuffer, m_fieldPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SlimeFieldPushConstants), &fieldPush);
                vkCmdDispatch(m_commandBuffer, fieldGroupsX, fieldGroupsY, fieldGroupsZ);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

                m_fieldParity ^= 1;
            }

            if (agentCount > 0)
            {
                VkMemoryBarrier readbackBarrier = {};
                readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);

                VkBufferCopy region = {};
                region.size = agentSize;
                vkCmdCopyBuffer(m_commandBuffer, m_agentBuffer, m_stagingBuffer, 1, &region);
            }

            SubmitAndWait();

            if (agentCount > 0)
            {
                std::memcpy(agents.data(), m_pStagingMapped, static_cast<size_t>(agentSize));
            }
        }
    }
}
