#include "Hal/Gpu/GrayScott3DKernel.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct GrayScott3DPushConstants
        {
            float feed;
            float kill;
            float diffU;
            float diffV;
            int32_t xDim;
            int32_t yDim;
            int32_t zDim;
        };

        // GrayScott3D.comp runs an 8x8x8 workgroup over the grid, one
        // invocation per voxel; each voxel is a vec2 (U, V).
        const uint32_t g_grayScott3DGroupSize = 8;

        GrayScott3DKernel::GrayScott3DKernel(ComputeContext &context,
                                             int32_t         xDim,
                                             int32_t         yDim,
                                             int32_t         zDim)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_zDim(zDim),
              m_length(xDim * yDim * zDim),
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
              m_descriptorSetAToB(VK_NULL_HANDLE),
              m_descriptorSetBToA(VK_NULL_HANDLE),
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

        GrayScott3DKernel::~GrayScott3DKernel()
        {
            Destroy();
        }

        void GrayScott3DKernel::Destroy()
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

        void GrayScott3DKernel::CreateBuffers()
        {
            // Two floats (U, V) per voxel.
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * 2 * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_bufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationA);
            m_bufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationB);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void GrayScott3DKernel::CreateDescriptorObjects()
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
            m_descriptorSetAToB = sets[0];
            m_descriptorSetBToA = sets[1];

            VkDescriptorBufferInfo infoA = {};
            infoA.buffer = m_bufferA;
            infoA.offset = 0;
            infoA.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo infoB = {};
            infoB.buffer = m_bufferB;
            infoB.offset = 0;
            infoB.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writes[4] = {};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = m_descriptorSetAToB;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[0].pBufferInfo = &infoA;
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = m_descriptorSetAToB;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &infoB;
            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = m_descriptorSetBToA;
            writes[2].dstBinding = 0;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].pBufferInfo = &infoB;
            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = m_descriptorSetBToA;
            writes[3].dstBinding = 1;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].pBufferInfo = &infoA;

            vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        }

        void GrayScott3DKernel::CreatePipeline()
        {
            VkDevice device = m_context.Device();

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("GrayScott3D.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &m_shaderModule), "vkCreateShaderModule failed");

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(GrayScott3DPushConstants);

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

        void GrayScott3DKernel::CreateCommandObjects()
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

        void GrayScott3DKernel::SubmitAndWait()
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

        void GrayScott3DKernel::RunSteps(std::vector<float> &u,
                                         std::vector<float> &v,
                                         int32_t             steps,
                                         float               feed,
                                         float               kill,
                                         float               diffU,
                                         float               diffV)
        {
            Internal::CheckSize(u.size(), static_cast<size_t>(m_length), "GrayScott3DKernel::RunSteps u");
            Internal::CheckSize(v.size(), static_cast<size_t>(m_length), "GrayScott3DKernel::RunSteps v");
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * 2 * sizeof(float);

            float *pStaging = static_cast<float *>(m_pStagingMapped);
            for (int32_t i = 0; i < m_length; i++)
            {
                pStaging[2 * i] = u[static_cast<size_t>(i)];
                pStaging[2 * i + 1] = v[static_cast<size_t>(i)];
            }

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_bufferA, 1, &region);

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

            GrayScott3DPushConstants pushConstants = {};
            pushConstants.feed = feed;
            pushConstants.kill = kill;
            pushConstants.diffU = diffU;
            pushConstants.diffV = diffV;
            pushConstants.xDim = m_xDim;
            pushConstants.yDim = m_yDim;
            pushConstants.zDim = m_zDim;
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GrayScott3DPushConstants), &pushConstants);

            uint32_t groupCountX = (static_cast<uint32_t>(m_xDim) + g_grayScott3DGroupSize - 1) / g_grayScott3DGroupSize;
            uint32_t groupCountY = (static_cast<uint32_t>(m_yDim) + g_grayScott3DGroupSize - 1) / g_grayScott3DGroupSize;
            uint32_t groupCountZ = (static_cast<uint32_t>(m_zDim) + g_grayScott3DGroupSize - 1) / g_grayScott3DGroupSize;

            for (int32_t step = 0; step < steps; step++)
            {
                VkDescriptorSet set = ((step % 2) == 0) ? m_descriptorSetAToB : m_descriptorSetBToA;
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &set, 0, nullptr);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
            }

            VkBuffer resultBuffer = ((steps % 2) == 0) ? m_bufferA : m_bufferB;

            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(m_commandBuffer, resultBuffer, m_stagingBuffer, 1, &region);

            SubmitAndWait();

            for (int32_t i = 0; i < m_length; i++)
            {
                u[static_cast<size_t>(i)] = pStaging[2 * i];
                v[static_cast<size_t>(i)] = pStaging[2 * i + 1];
            }
        }
    }
}
