#include "Hal/Gpu/AdiKernel2D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"
#include "Hal/Log/Logger.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace Hal
{
    namespace Gpu
    {
        struct AdiPushConstants
        {
            float diffRate;
            int32_t xDim;
            int32_t yDim;
            int32_t axis;
            int32_t wrapSolve;
            int32_t wrapRhs;
        };

        // Shared-memory footprint per workgroup: the five float arrays
        // AdiSweep2D.comp declares at MAX_LINE length.
        const uint32_t g_adiWorkgroupSize = 1024;
        const uint32_t g_adiSharedBytes = 5 * 1024 * sizeof(float);

        AdiKernel2D::AdiKernel2D(ComputeContext &context,
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
              m_descriptorSetAToB(VK_NULL_HANDLE),
              m_descriptorSetBToA(VK_NULL_HANDLE),
              m_commandPool(VK_NULL_HANDLE),
              m_commandBuffer(VK_NULL_HANDLE),
              m_fence(VK_NULL_HANDLE)
        {
            try
            {
                if ((xDim > kMaxLine) || (yDim > kMaxLine))
                {
                    Logger::Instance().Log(LogSeverity::Error, "AdiKernel2D: grid dimensions exceed the per-workgroup line cap of " + std::to_string(kMaxLine));
                    throw std::invalid_argument("AdiKernel2D: grid dimension exceeds the PCR line cap");
                }

                if (!IsSupported(m_context))
                {
                    Logger::Instance().Log(LogSeverity::Error, "AdiKernel2D: device workgroup limits too small for the PCR solve (needs 1024 invocations and 20KB shared memory)");
                    throw std::runtime_error("AdiKernel2D: device workgroup limits too small for the PCR solve");
                }

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

        bool AdiKernel2D::IsSupported(ComputeContext &context)
        {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(context.PhysicalDevice(), &properties);
            return (properties.limits.maxComputeWorkGroupInvocations >= g_adiWorkgroupSize) &&
                   (properties.limits.maxComputeSharedMemorySize >= g_adiSharedBytes);
        }

        AdiKernel2D::~AdiKernel2D()
        {
            Destroy();
        }

        void AdiKernel2D::Destroy()
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

        void AdiKernel2D::CreateBuffers()
        {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_bufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationA);
            m_bufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), size, deviceUsage, m_allocationB);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void AdiKernel2D::CreateDescriptorObjects()
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

        void AdiKernel2D::CreatePipeline()
        {
            VkDevice device = m_context.Device();

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath("AdiSweep2D.comp"));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &m_shaderModule), "vkCreateShaderModule failed");

            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(AdiPushConstants);

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

        void AdiKernel2D::CreateCommandObjects()
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

        void AdiKernel2D::SubmitAndWait()
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

        void AdiKernel2D::RunSteps(std::vector<float> &field,
                                   int32_t             steps,
                                   float               diffRate,
                                   bool                wrapX,
                                   bool                wrapY)
        {
            Internal::CheckSize(field.size(), static_cast<size_t>(m_length), "AdiKernel2D::RunSteps field");
            VkDeviceSize size = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            std::memcpy(m_pStagingMapped, field.data(), static_cast<size_t>(size));

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBufferCopy region = {};
            region.size = size;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, m_bufferA, 1, &region);

            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            VkMemoryBarrier computeBarrier = {};
            computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

            AdiPushConstants sweepX = {};
            sweepX.diffRate = diffRate;
            sweepX.xDim = m_xDim;
            sweepX.yDim = m_yDim;
            sweepX.axis = 0;
            sweepX.wrapSolve = wrapX ? 1 : 0;
            sweepX.wrapRhs = wrapY ? 1 : 0;

            AdiPushConstants sweepY = {};
            sweepY.diffRate = diffRate;
            sweepY.xDim = m_xDim;
            sweepY.yDim = m_yDim;
            sweepY.axis = 1;
            sweepY.wrapSolve = wrapY ? 1 : 0;
            sweepY.wrapRhs = wrapX ? 1 : 0;

            for (int32_t step = 0; step < steps; step++)
            {
                // Sweep 1: solve along x for every row, A -> B.
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSetAToB, 0, nullptr);
                vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdiPushConstants), &sweepX);
                vkCmdDispatch(m_commandBuffer, static_cast<uint32_t>(m_yDim), 1, 1);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

                // Sweep 2: solve along y for every column, B -> A.
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorSetBToA, 0, nullptr);
                vkCmdPushConstants(m_commandBuffer, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdiPushConstants), &sweepY);
                vkCmdDispatch(m_commandBuffer, static_cast<uint32_t>(m_xDim), 1, 1);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
            }

            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(m_commandBuffer, m_bufferA, m_stagingBuffer, 1, &region);

            SubmitAndWait();
            std::memcpy(field.data(), m_pStagingMapped, static_cast<size_t>(size));
        }
    }
}
