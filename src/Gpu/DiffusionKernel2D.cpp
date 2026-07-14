#include "Hal/Gpu/DiffusionKernel2D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct DiffusionPushConstants
        {
            float diffRate;
            int32_t xDim;
            int32_t yDim;
            int32_t wrapX;
            int32_t wrapY;
        };

        // Workgroup sizes fixed in the shaders: Diffusion2D.comp runs
        // 16x16 over the grid, DiffusionUpdate2D.comp runs 256 linear
        // over the flat cell array.
        const uint32_t g_diffusionGroupSize = 16;
        const uint32_t g_updateGroupSize = 256;

        static VkShaderModule CreateComputeShaderModule(VkDevice    device,
                                                        const char *pShaderName)
        {
            std::vector<char> code = Internal::ReadFile(ShaderPath(pShaderName));
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
            VkShaderModule shaderModule = VK_NULL_HANDLE;
            Internal::CheckResult(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule), "vkCreateShaderModule failed");
            return shaderModule;
        }

        // Full memory barrier between pipeline stages — the simple,
        // correct hammer for chaining copy and compute passes inside one
        // command buffer.
        static void RecordMemoryBarrier(VkCommandBuffer      commandBuffer,
                                        VkPipelineStageFlags srcStage,
                                        VkAccessFlags        srcAccess,
                                        VkPipelineStageFlags dstStage,
                                        VkAccessFlags        dstAccess)
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = srcAccess;
            barrier.dstAccessMask = dstAccess;
            vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        DiffusionKernel2D::DiffusionKernel2D(ComputeContext &context,
                                             int32_t         xDim,
                                             int32_t         yDim)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_length(xDim * yDim),
              m_fieldBuffer(VK_NULL_HANDLE),
              m_fieldAllocation(VK_NULL_HANDLE),
              m_deltaBuffer(VK_NULL_HANDLE),
              m_deltaAllocation(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_shaderModule(VK_NULL_HANDLE),
              m_updateShaderModule(VK_NULL_HANDLE),
              m_descriptorSetLayout(VK_NULL_HANDLE),
              m_pipelineLayout(VK_NULL_HANDLE),
              m_pipeline(VK_NULL_HANDLE),
              m_updatePipeline(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_descriptorSet(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_commandBuffer(VK_NULL_HANDLE),
              m_fence(VK_NULL_HANDLE)
        {
            CreateBuffers();
            CreateDescriptorObjects();
            CreatePipelines();
            CreateCommandObjects();
            ClearDeltas();
        }

        DiffusionKernel2D::~DiffusionKernel2D()
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
            if (m_pipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_pipeline, nullptr);
            }
            if (m_updatePipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_updatePipeline, nullptr);
            }
            if (m_pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
            }
            if (m_shaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_shaderModule, nullptr);
            }
            if (m_updateShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_updateShaderModule, nullptr);
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
            Internal::DestroyBufferVma(m_context.Allocator(), m_deltaBuffer, m_deltaAllocation);
            Internal::DestroyBufferVma(m_context.Allocator(), m_fieldBuffer, m_fieldAllocation);
        }

        void DiffusionKernel2D::CreateBuffers()
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_fieldBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_fieldAllocation);
            m_deltaBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_deltaAllocation);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void DiffusionKernel2D::SubmitAndWait()
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

        void DiffusionKernel2D::UploadField(const std::vector<float> &field)
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            std::memcpy(m_pStagingMapped, field.data(), static_cast<size_t>(size));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_fieldBuffer, 1, &region);
            SubmitAndWait();
        }

        void DiffusionKernel2D::ClearDeltas()
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            vkCmdFillBuffer(m_commandBuffer, m_deltaBuffer, 0, VK_WHOLE_SIZE, 0);
            SubmitAndWait();
        }

        void DiffusionKernel2D::Dispatch(float  diffRate,
                                         bool   wrapX,
                                         bool   wrapY)
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

            DiffusionPushConstants pushConstants = {};
            pushConstants.diffRate = diffRate;
            pushConstants.xDim = m_xDim;
            pushConstants.yDim = m_yDim;
            pushConstants.wrapX = wrapX ? 1 : 0;
            pushConstants.wrapY = wrapY ? 1 : 0;
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DiffusionPushConstants), &pushConstants);

            uint32_t groupCountX = (static_cast<uint32_t>(m_xDim) + g_diffusionGroupSize - 1) / g_diffusionGroupSize;
            uint32_t groupCountY = (static_cast<uint32_t>(m_yDim) + g_diffusionGroupSize - 1) / g_diffusionGroupSize;
            vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, 1);

            // Readback copy in the same submission: deltas land in the
            // mapped staging buffer for DownloadDeltas to read directly.
            RecordMemoryBarrier(m_commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferCopy region = {};
            region.size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            vkCmdCopyBuffer(m_commandBuffer, m_deltaBuffer, m_stagingBuffer, 1, &region);

            SubmitAndWait();
        }

        std::vector<float> DiffusionKernel2D::DownloadDeltas() const
        {
            std::vector<float> result(static_cast<size_t>(m_length));
            std::memcpy(result.data(), m_pStagingMapped, static_cast<size_t>(m_length) * sizeof(float));
            return result;
        }

        void DiffusionKernel2D::RunSteps(std::vector<float> &field,
                                         int32_t             steps,
                                         float               diffRate,
                                         bool                wrapX,
                                         bool                wrapY)
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            std::memcpy(m_pStagingMapped, field.data(), static_cast<size_t>(size));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            // Upload + clear once, then chain the steps GPU-resident.
            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_fieldBuffer, 1, &region);
            vkCmdFillBuffer(m_commandBuffer, m_deltaBuffer, 0, VK_WHOLE_SIZE, 0);
            RecordMemoryBarrier(m_commandBuffer,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

            DiffusionPushConstants pushConstants = {};
            pushConstants.diffRate = diffRate;
            pushConstants.xDim = m_xDim;
            pushConstants.yDim = m_yDim;
            pushConstants.wrapX = wrapX ? 1 : 0;
            pushConstants.wrapY = wrapY ? 1 : 0;
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DiffusionPushConstants), &pushConstants);

            uint32_t groupCountX = (static_cast<uint32_t>(m_xDim) + g_diffusionGroupSize - 1) / g_diffusionGroupSize;
            uint32_t groupCountY = (static_cast<uint32_t>(m_yDim) + g_diffusionGroupSize - 1) / g_diffusionGroupSize;
            uint32_t updateGroupCount = (static_cast<uint32_t>(m_length) + g_updateGroupSize - 1) / g_updateGroupSize;

            for (int32_t step = 0; step < steps; step++)
            {
                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, 1);
                RecordMemoryBarrier(m_commandBuffer,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_updatePipeline);
                vkCmdDispatch(m_commandBuffer, updateGroupCount, 1, 1);
                RecordMemoryBarrier(m_commandBuffer,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            }

            RecordMemoryBarrier(m_commandBuffer,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyBuffer(m_commandBuffer, m_fieldBuffer, m_stagingBuffer, 1, &region);

            SubmitAndWait();
            std::memcpy(field.data(), m_pStagingMapped, static_cast<size_t>(size));
        }

        void DiffusionKernel2D::CreateDescriptorObjects()
        {
            VkDevice device = m_context.Device();

            VkDescriptorSetLayoutBinding bindings[2] = {};
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
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 2;

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 1;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = m_descriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_descriptorSetLayout;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, &m_descriptorSet), "vkAllocateDescriptorSets failed");

            VkDescriptorBufferInfo fieldInfo = {};
            fieldInfo.buffer = m_fieldBuffer;
            fieldInfo.offset = 0;
            fieldInfo.range = VK_WHOLE_SIZE;

            VkDescriptorBufferInfo deltaInfo = {};
            deltaInfo.buffer = m_deltaBuffer;
            deltaInfo.offset = 0;
            deltaInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[2] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = m_descriptorSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &fieldInfo;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = m_descriptorSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &deltaInfo;

            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        }

        void DiffusionKernel2D::CreatePipelines()
        {
            VkDevice device = m_context.Device();

            m_shaderModule = CreateComputeShaderModule(device, "Diffusion2D.comp");
            m_updateShaderModule = CreateComputeShaderModule(device, "DiffusionUpdate2D.comp");

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(DiffusionPushConstants);

            VkPipelineLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &m_descriptorSetLayout;
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pushConstantRange;
            Internal::CheckResult(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout), "vkCreatePipelineLayout failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = m_shaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = m_pipelineLayout;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline), "vkCreateComputePipelines failed");

            // The update pass shares the descriptor set and pipeline
            // layout; only the shader differs.
            stageInfo.module = m_updateShaderModule;
            pipelineInfo.stage = stageInfo;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_updatePipeline), "vkCreateComputePipelines (update) failed");
        }

        void DiffusionKernel2D::CreateCommandObjects()
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
