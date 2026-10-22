#include "Hal/Gpu/WaveKernel2D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct WavePushConstants
        {
            float speed2;
            float damping;
            int32_t xDim;
            int32_t yDim;
            int32_t wrap;
        };

        // Wave2D.comp runs a 16x16 workgroup over the grid, one invocation
        // per cell.
        const uint32_t g_waveGroupSize = 16;

        WaveKernel2D::WaveKernel2D(ComputeContext &context,
                                   int32_t         xDim,
                                   int32_t         yDim)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_length(xDim * yDim),
              m_bufferA(VK_NULL_HANDLE),
              m_allocationA(VK_NULL_HANDLE),
              m_bufferB(VK_NULL_HANDLE),
              m_allocationB(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_shaderModule(VK_NULL_HANDLE),
              m_descriptorSetLayout(VK_NULL_HANDLE),
              m_pipelineLayout(VK_NULL_HANDLE),
              m_pipeline(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_descriptorSetAcurBprev(VK_NULL_HANDLE),
              m_descriptorSetBcurAprev(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_commandBuffer(VK_NULL_HANDLE),
              m_fence(VK_NULL_HANDLE)
        {
            try
            {
                CreateBuffers();
                CreateDescriptorObjects();
                CreatePipeline();
                CreateCommandObjects();
            }
            catch (...)
            {
                Destroy();
                throw;
            }
        }

        WaveKernel2D::~WaveKernel2D()
        {
            Destroy();
        }

        void WaveKernel2D::Destroy()
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
            if (m_pipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
            }
            if (m_shaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_shaderModule, nullptr);
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
            Internal::DestroyBufferVma(m_context.Allocator(), m_bufferB, m_allocationB);
            Internal::DestroyBufferVma(m_context.Allocator(), m_bufferA, m_allocationA);
        }

        void WaveKernel2D::CreateBuffers()
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_bufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationA);
            m_bufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationB);
            // Staging holds both time levels: [current | previous].
            VkDeviceSize stagingSize = static_cast<VkDeviceSize>(m_length) * 2 * sizeof(float);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void WaveKernel2D::CreateDescriptorObjects()
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
            poolSize.descriptorCount = 4;

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 2;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetLayout setLayouts[2] = {m_descriptorSetLayout, m_descriptorSetLayout};
            VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = m_descriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, sets), "vkAllocateDescriptorSets failed");
            m_descriptorSetAcurBprev = sets[0];
            m_descriptorSetBcurAprev = sets[1];

            VkDescriptorBufferInfo infoA = {};
            infoA.buffer = m_bufferA;
            infoA.offset = 0;
            infoA.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo infoB = {};
            infoB.buffer = m_bufferB;
            infoB.offset = 0;
            infoB.range = VK_WHOLE_SIZE;

            // Set 0: current = A (binding 0), previous/next = B (binding 1).
            // Set 1: current = B (binding 0), previous/next = A (binding 1).
            VkWriteDescriptorSet writes[4] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = m_descriptorSetAcurBprev;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &infoA;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = m_descriptorSetAcurBprev;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &infoB;
            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = m_descriptorSetBcurAprev;
            writes[2].dstBinding = 0;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].pBufferInfo = &infoB;
            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = m_descriptorSetBcurAprev;
            writes[3].dstBinding = 1;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].pBufferInfo = &infoA;

            vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        }

        void WaveKernel2D::CreatePipeline()
        {
            VkDevice device = m_context.Device();

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("Wave2D.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &m_shaderModule), "vkCreateShaderModule failed");

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(WavePushConstants);

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
        }

        void WaveKernel2D::CreateCommandObjects()
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

        void WaveKernel2D::SubmitAndWait()
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

        void WaveKernel2D::RunSteps(std::vector<float> &cur,
                                    std::vector<float> &prev,
                                    int32_t             steps,
                                    float               speed2,
                                    float               damping,
                                    bool                wrap)
        {
            Internal::CheckSize(cur.size(), static_cast<size_t>(m_length), "WaveKernel2D::RunSteps cur");
            Internal::CheckSize(prev.size(), static_cast<size_t>(m_length), "WaveKernel2D::RunSteps prev");
            VkDeviceSize bufSize = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            float *pStaging = static_cast<float *>(m_pStagingMapped);
            std::memcpy(pStaging, cur.data(), static_cast<size_t>(bufSize));
            std::memcpy(pStaging + m_length, prev.data(), static_cast<size_t>(bufSize));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            // Upload current -> A, previous -> B.
            VkBufferCopy toCur = {};
            toCur.srcOffset = 0;
            toCur.size = bufSize;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_bufferA, 1, &toCur);
            VkBufferCopy toPrev = {};
            toPrev.srcOffset = bufSize;
            toPrev.size = bufSize;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_bufferB, 1, &toPrev);

            VkMemoryBarrier uploadBarrier = {};
            uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploadBarrier, 0, nullptr, 0, nullptr);

            VkMemoryBarrier computeBarrier = {};
            computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

            WavePushConstants pushConstants = {};
            pushConstants.speed2 = speed2;
            pushConstants.damping = damping;
            pushConstants.xDim = m_xDim;
            pushConstants.yDim = m_yDim;
            pushConstants.wrap = wrap ? 1 : 0;
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(WavePushConstants), &pushConstants);

            uint32_t groupCountX = (static_cast<uint32_t>(m_xDim) + g_waveGroupSize - 1) / g_waveGroupSize;
            uint32_t groupCountY = (static_cast<uint32_t>(m_yDim) + g_waveGroupSize - 1) / g_waveGroupSize;

            // Each step reads the current level and overwrites the previous
            // level with the next; the roles swap, so alternate descriptor
            // sets. On even steps the current level is in A, on odd in B.
            for (int32_t step = 0; step < steps; step++)
            {
                VkDescriptorSet set = ((step % 2) == 0) ? m_descriptorSetAcurBprev : m_descriptorSetBcurAprev;
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &set, 0, nullptr);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, 1);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
            }

            // After `steps` steps the current level is in A if steps is even
            // (last write was into A), in B if odd; the previous level is
            // the other buffer.
            VkBuffer curBuffer = ((steps % 2) == 0) ? m_bufferA : m_bufferB;
            VkBuffer prevBuffer = ((steps % 2) == 0) ? m_bufferB : m_bufferA;

            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);

            VkBufferCopy fromCur = {};
            fromCur.dstOffset = 0;
            fromCur.size = bufSize;
            vkCmdCopyBuffer(m_commandBuffer, curBuffer, m_stagingBuffer, 1, &fromCur);
            VkBufferCopy fromPrev = {};
            fromPrev.dstOffset = bufSize;
            fromPrev.size = bufSize;
            vkCmdCopyBuffer(m_commandBuffer, prevBuffer, m_stagingBuffer, 1, &fromPrev);

            SubmitAndWait();

            std::memcpy(cur.data(), pStaging, static_cast<size_t>(bufSize));
            std::memcpy(prev.data(), pStaging + m_length, static_cast<size_t>(bufSize));
        }
    }
}
