#include "Hal/Gpu/FlowFieldKernel3D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct FlowFieldPushConstants
        {
            float coeffA;
            float coeffB;
            float coeffC;
            float dt;
            int32_t count;
        };

        // FlowField3D.comp runs a linear workgroup of 256 over the particle
        // array.
        const uint32_t g_flowFieldGroupSize = 256;

        FlowFieldKernel3D::FlowFieldKernel3D(ComputeContext &context,
                                             int32_t         particleCount)
            : m_context(context),
              m_particleCount(particleCount),
              m_positionBuffer(VK_NULL_HANDLE),
              m_positionAllocation(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_shaderModule(VK_NULL_HANDLE),
              m_descriptorSetLayout(VK_NULL_HANDLE),
              m_pipelineLayout(VK_NULL_HANDLE),
              m_pipeline(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_descriptorSet(VK_NULL_HANDLE),
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

        FlowFieldKernel3D::~FlowFieldKernel3D()
        {
            Destroy();
        }

        void FlowFieldKernel3D::Destroy()
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
            Internal::DestroyBufferVma(m_context.Allocator(), m_positionBuffer, m_positionAllocation);
        }

        void FlowFieldKernel3D::CreateBuffers()
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_particleCount) * 4 * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_positionBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_positionAllocation);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void FlowFieldKernel3D::CreateDescriptorObjects()
        {
            VkDevice device = m_context.Device();

            VkDescriptorSetLayoutBinding binding = {};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings = &binding;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_descriptorSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 1;

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

            VkDescriptorBufferInfo info = {};
            info.buffer = m_positionBuffer;
            info.offset = 0;
            info.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet write = {};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = m_descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void FlowFieldKernel3D::CreatePipeline()
        {
            VkDevice device = m_context.Device();

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("FlowField3D.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &m_shaderModule), "vkCreateShaderModule failed");

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(FlowFieldPushConstants);

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

        void FlowFieldKernel3D::CreateCommandObjects()
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

        void FlowFieldKernel3D::SubmitAndWait()
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

        void FlowFieldKernel3D::RunSteps(std::vector<float> &positions,
                                         int32_t             steps,
                                         float               a,
                                         float               b,
                                         float               c,
                                         float               dt)
        {
            Internal::CheckSize(positions.size(), static_cast<size_t>(m_particleCount) * 4, "FlowFieldKernel3D::RunSteps positions");
            VkDeviceSize size = static_cast<VkDeviceSize>(m_particleCount) * 4 * sizeof(float);
            std::memcpy(m_pStagingMapped, positions.data(), static_cast<size_t>(size));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_positionBuffer, 1, &region);

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
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

            FlowFieldPushConstants pushConstants = {};
            pushConstants.coeffA = a;
            pushConstants.coeffB = b;
            pushConstants.coeffC = c;
            pushConstants.dt = dt;
            pushConstants.count = m_particleCount;
            vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FlowFieldPushConstants), &pushConstants);

            uint32_t groupCount = (static_cast<uint32_t>(m_particleCount) + g_flowFieldGroupSize - 1) / g_flowFieldGroupSize;
            for (int32_t step = 0; step < steps; step++)
            {
                vkCmdDispatch(m_commandBuffer, groupCount, 1, 1);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
            }

            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(m_commandBuffer, m_positionBuffer, m_stagingBuffer, 1, &region);

            SubmitAndWait();
            std::memcpy(positions.data(), m_pStagingMapped, static_cast<size_t>(size));
        }
    }
}
