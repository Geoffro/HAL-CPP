#include "Hal/Gpu/AgentForceKernel2D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct AgentPushConstants
        {
            float interactionRad;
            float springConstant;
            int32_t numAgents;
            int32_t xDim;
            int32_t yDim;
            int32_t wrapX;
            int32_t wrapY;
        };

        AgentForceKernel2D::AgentForceKernel2D(ComputeContext &context,
                                               int32_t         xDim,
                                               int32_t         yDim,
                                               int32_t         maxAgents)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_numCells(xDim * yDim),
              m_maxAgents(maxAgents),
              m_numAgents(0),
              m_positionBuffer(VK_NULL_HANDLE),
              m_positionAllocation(VK_NULL_HANDLE),
              m_radiusBuffer(VK_NULL_HANDLE),
              m_radiusAllocation(VK_NULL_HANDLE),
              m_cellCountBuffer(VK_NULL_HANDLE),
              m_cellCountAllocation(VK_NULL_HANDLE),
              m_cellOffsetBuffer(VK_NULL_HANDLE),
              m_cellOffsetAllocation(VK_NULL_HANDLE),
              m_cellCursorBuffer(VK_NULL_HANDLE),
              m_cellCursorAllocation(VK_NULL_HANDLE),
              m_bucketAgentBuffer(VK_NULL_HANDLE),
              m_bucketAgentAllocation(VK_NULL_HANDLE),
              m_velocityDeltaBuffer(VK_NULL_HANDLE),
              m_velocityDeltaAllocation(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_cellCountPipeline{},
              m_prefixSumPipeline{},
              m_scatterPipeline{},
              m_forceAccumPipeline{},
              m_commandPool(VK_NULL_HANDLE),
              m_commandBuffer(VK_NULL_HANDLE),
              m_fence(VK_NULL_HANDLE)
        {
            try
            {
                CreateBuffers();
                CreateCellCountPipeline();
                CreatePrefixSumPipeline();
                CreateScatterPipeline();
                CreateForceAccumPipeline();
                CreateCommandObjects();
            }
            catch (...)
            {
                Destroy();
                throw;
            }
        }

        AgentForceKernel2D::~AgentForceKernel2D()
        {
            Destroy();
        }

        void AgentForceKernel2D::Destroy()
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

            DestroyPipelineObjects(m_forceAccumPipeline);
            DestroyPipelineObjects(m_scatterPipeline);
            DestroyPipelineObjects(m_prefixSumPipeline);
            DestroyPipelineObjects(m_cellCountPipeline);

            VmaAllocator allocator = m_context.Allocator();
            Internal::DestroyBufferVma(allocator, m_stagingBuffer, m_stagingAllocation);
            Internal::DestroyBufferVma(allocator, m_velocityDeltaBuffer, m_velocityDeltaAllocation);
            Internal::DestroyBufferVma(allocator, m_bucketAgentBuffer, m_bucketAgentAllocation);
            Internal::DestroyBufferVma(allocator, m_cellCursorBuffer, m_cellCursorAllocation);
            Internal::DestroyBufferVma(allocator, m_cellOffsetBuffer, m_cellOffsetAllocation);
            Internal::DestroyBufferVma(allocator, m_cellCountBuffer, m_cellCountAllocation);
            Internal::DestroyBufferVma(allocator, m_radiusBuffer, m_radiusAllocation);
            Internal::DestroyBufferVma(allocator, m_positionBuffer, m_positionAllocation);
        }

        void AgentForceKernel2D::DestroyPipelineObjects(PipelineObjects &objects)
        {
            VkDevice device = m_context.Device();
            if (objects.descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, objects.descriptorPool, nullptr);
            }
            if (objects.pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, objects.pipeline, nullptr);
            }
            if (objects.pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, objects.pipelineLayout, nullptr);
            }
            if (objects.descriptorSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, objects.descriptorSetLayout, nullptr);
            }
            if (objects.shaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, objects.shaderModule, nullptr);
            }
        }

        void AgentForceKernel2D::CreateBuffers()
        {
            VkDeviceSize agentVecSize = static_cast<VkDeviceSize>(m_maxAgents) * 2 * sizeof(float);
            VkDeviceSize agentScalarSize = static_cast<VkDeviceSize>(m_maxAgents) * sizeof(float);
            VkDeviceSize cellIntSize = static_cast<VkDeviceSize>(m_numCells) * sizeof(int32_t);
            VkDeviceSize cellOffsetSize = static_cast<VkDeviceSize>(m_numCells + 1) * sizeof(int32_t);
            VkDeviceSize agentIntSize = static_cast<VkDeviceSize>(m_maxAgents) * sizeof(int32_t);
            VmaAllocator allocator = m_context.Allocator();
            VkBufferUsageFlags storageDst = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_positionBuffer = Internal::CreateDeviceBufferVma(allocator, agentVecSize, storageDst, m_positionAllocation);
            m_radiusBuffer = Internal::CreateDeviceBufferVma(allocator, agentScalarSize, storageDst, m_radiusAllocation);
            m_cellCountBuffer = Internal::CreateDeviceBufferVma(allocator, cellIntSize, storageDst, m_cellCountAllocation);
            m_cellOffsetBuffer = Internal::CreateDeviceBufferVma(allocator, cellOffsetSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_cellOffsetAllocation);
            m_cellCursorBuffer = Internal::CreateDeviceBufferVma(allocator, cellIntSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_cellCursorAllocation);
            m_bucketAgentBuffer = Internal::CreateDeviceBufferVma(allocator, agentIntSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, m_bucketAgentAllocation);
            m_velocityDeltaBuffer = Internal::CreateDeviceBufferVma(allocator, agentVecSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, m_velocityDeltaAllocation);

            // [positions | radii | velocity readback] — see the header.
            VkDeviceSize stagingSize = agentVecSize + agentScalarSize + agentVecSize;
            m_stagingBuffer = Internal::CreateStagingBufferVma(allocator, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void AgentForceKernel2D::SubmitAndWait()
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

        void AgentForceKernel2D::UploadAgents(const std::vector<float> &positions,
                                              const std::vector<float> &radii,
                                              int32_t                   numAgents)
        {
            Internal::CheckArgument((numAgents >= 0) && (numAgents <= m_maxAgents), "AgentForceKernel2D::UploadAgents numAgents must be within [0, maxAgents]");
            Internal::CheckArgument(positions.size() >= (static_cast<size_t>(numAgents) * 2), "AgentForceKernel2D::UploadAgents positions must hold 2 floats per agent");
            Internal::CheckArgument(radii.size() >= static_cast<size_t>(numAgents), "AgentForceKernel2D::UploadAgents radii must hold 1 float per agent");
            // Only writes staging; Dispatch() records the device copies.
            m_numAgents = numAgents;

            char *pStaging = static_cast<char *>(m_pStagingMapped);
            size_t radStagingOffset = static_cast<size_t>(m_maxAgents) * 2 * sizeof(float);
            std::memcpy(pStaging, positions.data(), static_cast<size_t>(numAgents) * 2 * sizeof(float));
            std::memcpy(pStaging + radStagingOffset, radii.data(), static_cast<size_t>(numAgents) * sizeof(float));
        }

        void AgentForceKernel2D::Dispatch(float  interactionRad,
                                          float  springConstant,
                                          bool   wrapX,
                                          bool   wrapY)
        {
            if (m_numAgents == 0)
            {
                return;
            }

            struct GridPushConstants
            {
                int32_t numAgents;
                int32_t xDim;
                int32_t yDim;
            };
            GridPushConstants gridConstants = {m_numAgents, m_xDim, m_yDim};

            struct PrefixSumPushConstants
            {
                int32_t numCells;
            };
            PrefixSumPushConstants prefixConstants = {m_numCells};

            AgentPushConstants forceConstants = {};
            forceConstants.interactionRad = interactionRad;
            forceConstants.springConstant = springConstant;
            forceConstants.numAgents = m_numAgents;
            forceConstants.xDim = m_xDim;
            forceConstants.yDim = m_yDim;
            forceConstants.wrapX = wrapX ? 1 : 0;
            forceConstants.wrapY = wrapY ? 1 : 0;

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            // Upload the staged agents and clear the cell counts inside
            // this same submission (UploadAgents only wrote the mapped
            // staging buffer).
            VkDeviceSize radStagingOffset = static_cast<VkDeviceSize>(m_maxAgents) * 2 * sizeof(float);
            VkBufferCopy posRegion = {};
            posRegion.size = static_cast<VkDeviceSize>(m_numAgents) * 2 * sizeof(float);
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_positionBuffer, 1, &posRegion);
            VkBufferCopy radRegion = {};
            radRegion.srcOffset = radStagingOffset;
            radRegion.size = static_cast<VkDeviceSize>(m_numAgents) * sizeof(float);
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_radiusBuffer, 1, &radRegion);
            vkCmdFillBuffer(m_commandBuffer, m_cellCountBuffer, 0, VK_WHOLE_SIZE, 0);

            VkMemoryBarrier uploadBarrier = {};
            uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploadBarrier, 0, nullptr, 0, nullptr);

            const uint32_t groupSize = 256;
            uint32_t agentGroups = (static_cast<uint32_t>(m_numAgents) + groupSize - 1) / groupSize;

            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_cellCountPipeline.pipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_cellCountPipeline.pipelineLayout, 0, 1, &m_cellCountPipeline.descriptorSet, 0, nullptr);
            vkCmdPushConstants(m_commandBuffer, m_cellCountPipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPushConstants), &gridConstants);
            vkCmdDispatch(m_commandBuffer, agentGroups, 1, 1);
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefixSumPipeline.pipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefixSumPipeline.pipelineLayout, 0, 1, &m_prefixSumPipeline.descriptorSet, 0, nullptr);
            vkCmdPushConstants(m_commandBuffer, m_prefixSumPipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PrefixSumPushConstants), &prefixConstants);
            vkCmdDispatch(m_commandBuffer, 1, 1, 1);
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_scatterPipeline.pipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_scatterPipeline.pipelineLayout, 0, 1, &m_scatterPipeline.descriptorSet, 0, nullptr);
            vkCmdPushConstants(m_commandBuffer, m_scatterPipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GridPushConstants), &gridConstants);
            vkCmdDispatch(m_commandBuffer, agentGroups, 1, 1);
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_forceAccumPipeline.pipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_forceAccumPipeline.pipelineLayout, 0, 1, &m_forceAccumPipeline.descriptorSet, 0, nullptr);
            vkCmdPushConstants(m_commandBuffer, m_forceAccumPipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AgentPushConstants), &forceConstants);
            vkCmdDispatch(m_commandBuffer, agentGroups, 1, 1);

            // Velocity readback into the staging buffer's third region,
            // still inside this one submission.
            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
            VkBufferCopy velRegion = {};
            velRegion.dstOffset = static_cast<VkDeviceSize>(m_maxAgents) * 3 * sizeof(float);
            velRegion.size = static_cast<VkDeviceSize>(m_numAgents) * 2 * sizeof(float);
            vkCmdCopyBuffer(m_commandBuffer, m_velocityDeltaBuffer, m_stagingBuffer, 1, &velRegion);

            SubmitAndWait();
        }

        std::vector<float> AgentForceKernel2D::DownloadVelocityDeltas() const
        {
            const char *pStaging = static_cast<const char *>(m_pStagingMapped);
            size_t velStagingOffset = static_cast<size_t>(m_maxAgents) * 3 * sizeof(float);
            std::vector<float> result(static_cast<size_t>(m_numAgents) * 2);
            std::memcpy(result.data(), pStaging + velStagingOffset, result.size() * sizeof(float));
            return result;
        }

        void AgentForceKernel2D::CreateCellCountPipeline()
        {
            VkDevice device = m_context.Device();
            PipelineObjects &objects = m_cellCountPipeline;

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("AgentCellCount.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &objects.shaderModule), "vkCreateShaderModule failed");

            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 2;
            layoutInfo.pBindings = bindings;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &objects.descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 2;
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &objects.descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = objects.descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &objects.descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, &objects.descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorBufferInfo positionInfo = {};
            positionInfo.buffer = m_positionBuffer;
            positionInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo cellCountInfo = {};
            cellCountInfo.buffer = m_cellCountBuffer;
            cellCountInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = objects.descriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &positionInfo;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = objects.descriptorSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &cellCountInfo;
            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.size = 3 * sizeof(int32_t);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &objects.descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &objects.pipelineLayout), "vkCreatePipelineLayout failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = objects.shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = objects.pipelineLayout;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &objects.pipeline), "vkCreateComputePipelines failed");
        }

        void AgentForceKernel2D::CreatePrefixSumPipeline()
        {
            VkDevice device = m_context.Device();
            PipelineObjects &objects = m_prefixSumPipeline;

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("AgentPrefixSum.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &objects.shaderModule), "vkCreateShaderModule failed");

            VkDescriptorSetLayoutBinding bindings[3]{};
            for (uint32_t i = 0; i < 3; i++)
            {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 3;
            layoutInfo.pBindings = bindings;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &objects.descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 3;
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &objects.descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = objects.descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &objects.descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, &objects.descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorBufferInfo cellCountInfo = {};
            cellCountInfo.buffer = m_cellCountBuffer;
            cellCountInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo cellOffsetInfo = {};
            cellOffsetInfo.buffer = m_cellOffsetBuffer;
            cellOffsetInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo cellCursorInfo = {};
            cellCursorInfo.buffer = m_cellCursorBuffer;
            cellCursorInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[3]{};
            VkDescriptorBufferInfo *pInfos[3] = {&cellCountInfo, &cellOffsetInfo, &cellCursorInfo};
            for (uint32_t i = 0; i < 3; i++)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = objects.descriptorSet;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = pInfos[i];
            }
            vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.size = sizeof(int32_t);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &objects.descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &objects.pipelineLayout), "vkCreatePipelineLayout failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = objects.shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = objects.pipelineLayout;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &objects.pipeline), "vkCreateComputePipelines failed");
        }

        void AgentForceKernel2D::CreateScatterPipeline()
        {
            VkDevice device = m_context.Device();
            PipelineObjects &objects = m_scatterPipeline;

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("AgentScatter.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &objects.shaderModule), "vkCreateShaderModule failed");

            VkDescriptorSetLayoutBinding bindings[3]{};
            for (uint32_t i = 0; i < 3; i++)
            {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 3;
            layoutInfo.pBindings = bindings;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &objects.descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 3;
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &objects.descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = objects.descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &objects.descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, &objects.descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorBufferInfo positionInfo = {};
            positionInfo.buffer = m_positionBuffer;
            positionInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo cellCursorInfo = {};
            cellCursorInfo.buffer = m_cellCursorBuffer;
            cellCursorInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo bucketAgentInfo = {};
            bucketAgentInfo.buffer = m_bucketAgentBuffer;
            bucketAgentInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[3]{};
            VkDescriptorBufferInfo *pInfos[3] = {&positionInfo, &cellCursorInfo, &bucketAgentInfo};
            for (uint32_t i = 0; i < 3; i++)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = objects.descriptorSet;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = pInfos[i];
            }
            vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.size = 3 * sizeof(int32_t);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &objects.descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &objects.pipelineLayout), "vkCreatePipelineLayout failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = objects.shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = objects.pipelineLayout;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &objects.pipeline), "vkCreateComputePipelines failed");
        }

        void AgentForceKernel2D::CreateForceAccumPipeline()
        {
            VkDevice device = m_context.Device();
            PipelineObjects &objects = m_forceAccumPipeline;

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("AgentForceAccum.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &objects.shaderModule), "vkCreateShaderModule failed");

            VkDescriptorSetLayoutBinding bindings[5]{};
            for (uint32_t i = 0; i < 5; i++)
            {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 5;
            layoutInfo.pBindings = bindings;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &objects.descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 5;
            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &objects.descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = objects.descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &objects.descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, &objects.descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorBufferInfo positionInfo = {};
            positionInfo.buffer = m_positionBuffer;
            positionInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo radiusInfo = {};
            radiusInfo.buffer = m_radiusBuffer;
            radiusInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo cellOffsetInfo = {};
            cellOffsetInfo.buffer = m_cellOffsetBuffer;
            cellOffsetInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo bucketAgentInfo = {};
            bucketAgentInfo.buffer = m_bucketAgentBuffer;
            bucketAgentInfo.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo velocityDeltaInfo = {};
            velocityDeltaInfo.buffer = m_velocityDeltaBuffer;
            velocityDeltaInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[5]{};
            VkDescriptorBufferInfo *pInfos[5] = {&positionInfo, &radiusInfo, &cellOffsetInfo, &bucketAgentInfo, &velocityDeltaInfo};
            for (uint32_t i = 0; i < 5; i++)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = objects.descriptorSet;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = pInfos[i];
            }
            vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.size = sizeof(AgentPushConstants);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &objects.descriptorSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &objects.pipelineLayout), "vkCreatePipelineLayout failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = objects.shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = objects.pipelineLayout;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &objects.pipeline), "vkCreateComputePipelines failed");
        }

        void AgentForceKernel2D::CreateCommandObjects()
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
    }
}
