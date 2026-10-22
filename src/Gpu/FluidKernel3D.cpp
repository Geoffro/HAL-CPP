#include "Hal/Gpu/FluidKernel3D.h"

#include "Hal/Gpu/Internal/VulkanUtil.h"
#include "Hal/Gpu/ShaderPaths.h"

#include "Hal/Gpu/Internal/VmaUsage.h"

#include <cstring>
#include <limits>

namespace Hal
{
    namespace Gpu
    {
        struct FluidAdvectPushConstants
        {
            int32_t xDim;
            int32_t yDim;
            int32_t zDim;
            float   dt;
            float   velocityDissipation;
            float   densityDissipation;
            float   temperatureDissipation;
            float   buoyancy;
            float   ambientTemperature;
        };

        struct FluidDivergencePushConstants
        {
            int32_t xDim;
            int32_t yDim;
            int32_t zDim;
            float   sourceX;
            float   sourceY;
            float   sourceZ;
            float   sourceRadius;
            float   sourceStrength;
        };

        struct FluidGridPushConstants
        {
            int32_t xDim;
            int32_t yDim;
            int32_t zDim;
        };

        // All four fluid shaders run an 8x8x8 workgroup over the grid,
        // one invocation per voxel.
        const uint32_t g_fluidGroupSize = 8;

        FluidKernel3D::FluidKernel3D(ComputeContext &context,
                                     int32_t         xDim,
                                     int32_t         yDim,
                                     int32_t         zDim)
            : m_context(context),
              m_xDim(xDim),
              m_yDim(yDim),
              m_zDim(zDim),
              m_length(xDim * yDim * zDim),
              m_parity(0),
              m_velBufferA(VK_NULL_HANDLE),
              m_velAllocationA(VK_NULL_HANDLE),
              m_velBufferB(VK_NULL_HANDLE),
              m_velAllocationB(VK_NULL_HANDLE),
              m_scalBufferA(VK_NULL_HANDLE),
              m_scalAllocationA(VK_NULL_HANDLE),
              m_scalBufferB(VK_NULL_HANDLE),
              m_scalAllocationB(VK_NULL_HANDLE),
              m_pressureBufferA(VK_NULL_HANDLE),
              m_pressureAllocationA(VK_NULL_HANDLE),
              m_pressureBufferB(VK_NULL_HANDLE),
              m_pressureAllocationB(VK_NULL_HANDLE),
              m_divergenceBuffer(VK_NULL_HANDLE),
              m_divergenceAllocation(VK_NULL_HANDLE),
              m_stagingBuffer(VK_NULL_HANDLE),
              m_stagingAllocation(VK_NULL_HANDLE),
              m_pStagingMapped(nullptr),
              m_advectShaderModule(VK_NULL_HANDLE),
              m_divergenceShaderModule(VK_NULL_HANDLE),
              m_jacobiShaderModule(VK_NULL_HANDLE),
              m_projectShaderModule(VK_NULL_HANDLE),
              m_advectSetLayout(VK_NULL_HANDLE),
              m_divergenceSetLayout(VK_NULL_HANDLE),
              m_jacobiSetLayout(VK_NULL_HANDLE),
              m_projectSetLayout(VK_NULL_HANDLE),
              m_advectPipelineLayout(VK_NULL_HANDLE),
              m_divergencePipelineLayout(VK_NULL_HANDLE),
              m_jacobiPipelineLayout(VK_NULL_HANDLE),
              m_projectPipelineLayout(VK_NULL_HANDLE),
              m_advectPipeline(VK_NULL_HANDLE),
              m_divergencePipeline(VK_NULL_HANDLE),
              m_jacobiPipeline(VK_NULL_HANDLE),
              m_projectPipeline(VK_NULL_HANDLE),
              m_descriptorPool(VK_NULL_HANDLE),
              m_advectSets(),
              m_divergenceSets(),
              m_jacobiSets(),
              m_projectSets(),
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
                ZeroInitBuffers();
            }
            catch (...)
            {
                Destroy();
                throw;
            }
        }

        FluidKernel3D::~FluidKernel3D()
        {
            Destroy();
        }

        void FluidKernel3D::Destroy()
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
            if (m_advectPipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_advectPipeline, nullptr);
            }
            if (m_divergencePipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_divergencePipeline, nullptr);
            }
            if (m_jacobiPipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_jacobiPipeline, nullptr);
            }
            if (m_projectPipeline != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(device, m_projectPipeline, nullptr);
            }
            if (m_advectPipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_advectPipelineLayout, nullptr);
            }
            if (m_divergencePipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_divergencePipelineLayout, nullptr);
            }
            if (m_jacobiPipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_jacobiPipelineLayout, nullptr);
            }
            if (m_projectPipelineLayout != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(device, m_projectPipelineLayout, nullptr);
            }
            if (m_advectShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_advectShaderModule, nullptr);
            }
            if (m_divergenceShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_divergenceShaderModule, nullptr);
            }
            if (m_jacobiShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_jacobiShaderModule, nullptr);
            }
            if (m_projectShaderModule != VK_NULL_HANDLE)
            {
                vkDestroyShaderModule(device, m_projectShaderModule, nullptr);
            }
            if (m_descriptorPool != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
            }
            if (m_advectSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, m_advectSetLayout, nullptr);
            }
            if (m_divergenceSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, m_divergenceSetLayout, nullptr);
            }
            if (m_jacobiSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, m_jacobiSetLayout, nullptr);
            }
            if (m_projectSetLayout != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(device, m_projectSetLayout, nullptr);
            }
            Internal::DestroyBufferVma(m_context.Allocator(), m_stagingBuffer, m_stagingAllocation);
            Internal::DestroyBufferVma(m_context.Allocator(), m_divergenceBuffer, m_divergenceAllocation);
            Internal::DestroyBufferVma(m_context.Allocator(), m_pressureBufferB, m_pressureAllocationB);
            Internal::DestroyBufferVma(m_context.Allocator(), m_pressureBufferA, m_pressureAllocationA);
            Internal::DestroyBufferVma(m_context.Allocator(), m_scalBufferB, m_scalAllocationB);
            Internal::DestroyBufferVma(m_context.Allocator(), m_scalBufferA, m_scalAllocationA);
            Internal::DestroyBufferVma(m_context.Allocator(), m_velBufferB, m_velAllocationB);
            Internal::DestroyBufferVma(m_context.Allocator(), m_velBufferA, m_velAllocationA);
        }

        void FluidKernel3D::CreateBuffers()
        {
            VkDeviceSize velSize = static_cast<VkDeviceSize>(m_length) * 4 * sizeof(float);
            VkDeviceSize scalSize = static_cast<VkDeviceSize>(m_length) * 2 * sizeof(float);
            VkDeviceSize floatSize = static_cast<VkDeviceSize>(m_length) * sizeof(float);
            VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            m_velBufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), velSize, deviceUsage, m_velAllocationA);
            m_velBufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), velSize, deviceUsage, m_velAllocationB);
            m_scalBufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), scalSize, deviceUsage, m_scalAllocationA);
            m_scalBufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), scalSize, deviceUsage, m_scalAllocationB);
            m_pressureBufferA = Internal::CreateDeviceBufferVma(m_context.Allocator(), floatSize, deviceUsage, m_pressureAllocationA);
            m_pressureBufferB = Internal::CreateDeviceBufferVma(m_context.Allocator(), floatSize, deviceUsage, m_pressureAllocationB);
            m_divergenceBuffer = Internal::CreateDeviceBufferVma(m_context.Allocator(), floatSize, deviceUsage, m_divergenceAllocation);
            m_stagingBuffer = Internal::CreateStagingBufferVma(m_context.Allocator(), scalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, m_stagingAllocation, &m_pStagingMapped);
        }

        void FluidKernel3D::CreateDescriptorObjects()
        {
            VkDevice device = m_context.Device();

            // One storage-buffer binding template, reused for every
            // layout (they differ only in binding count).
            VkDescriptorSetLayoutBinding bindings[4] = {};
            for (int32_t i = 0; i < 4; i++)
            {
                bindings[i].binding = static_cast<uint32_t>(i);
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.pBindings = bindings;

            layoutInfo.bindingCount = 4;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_advectSetLayout), "vkCreateDescriptorSetLayout failed");
            layoutInfo.bindingCount = 2;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_divergenceSetLayout), "vkCreateDescriptorSetLayout failed");
            layoutInfo.bindingCount = 3;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_jacobiSetLayout), "vkCreateDescriptorSetLayout failed");
            layoutInfo.bindingCount = 2;
            Internal::CheckResult(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_projectSetLayout), "vkCreateDescriptorSetLayout failed");

            VkDescriptorPoolSize poolSize = {};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = 22;

            VkDescriptorPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = 8;
            Internal::CheckResult(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool), "vkCreateDescriptorPool failed");

            VkDescriptorSetLayout setLayouts[8] = {
                m_advectSetLayout, m_advectSetLayout,
                m_divergenceSetLayout, m_divergenceSetLayout,
                m_jacobiSetLayout, m_jacobiSetLayout,
                m_projectSetLayout, m_projectSetLayout,
            };
            VkDescriptorSet sets[8] = {};
            VkDescriptorSetAllocateInfo setAllocInfo = {};
            setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAllocInfo.descriptorPool = m_descriptorPool;
            setAllocInfo.descriptorSetCount = 8;
            setAllocInfo.pSetLayouts = setLayouts;
            Internal::CheckResult(vkAllocateDescriptorSets(device, &setAllocInfo, sets), "vkAllocateDescriptorSets failed");
            m_advectSets[0] = sets[0];
            m_advectSets[1] = sets[1];
            m_divergenceSets[0] = sets[2];
            m_divergenceSets[1] = sets[3];
            m_jacobiSets[0] = sets[4];
            m_jacobiSets[1] = sets[5];
            m_projectSets[0] = sets[6];
            m_projectSets[1] = sets[7];

            VkDescriptorBufferInfo velA = {};
            velA.buffer = m_velBufferA;
            velA.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo velB = {};
            velB.buffer = m_velBufferB;
            velB.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo scalA = {};
            scalA.buffer = m_scalBufferA;
            scalA.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo scalB = {};
            scalB.buffer = m_scalBufferB;
            scalB.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo pressA = {};
            pressA.buffer = m_pressureBufferA;
            pressA.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo pressB = {};
            pressB.buffer = m_pressureBufferB;
            pressB.range = VK_WHOLE_SIZE;
            VkDescriptorBufferInfo diver = {};
            diver.buffer = m_divergenceBuffer;
            diver.range = VK_WHOLE_SIZE;

            // Parity 0: state is in the A buffers, the step writes the B
            // buffers (and projects velB in place). Parity 1 is the
            // mirror image. Jacobi always starts reading pressure A, and
            // an even iteration count lands the result back in A, which
            // is what both project sets read.
            const VkDescriptorBufferInfo *pBufferInfos[22] = {
                // advect set 0: velIn=A, velOut=B, scalIn=A, scalOut=B
                &velA, &velB, &scalA, &scalB,
                // advect set 1: velIn=B, velOut=A, scalIn=B, scalOut=A
                &velB, &velA, &scalB, &scalA,
                // divergence set 0: velIn=B (just written), divOut
                &velB, &diver,
                // divergence set 1: velIn=A, divOut
                &velA, &diver,
                // jacobi set 0: pIn=A, pOut=B, div
                &pressA, &pressB, &diver,
                // jacobi set 1: pIn=B, pOut=A, div
                &pressB, &pressA, &diver,
                // project set 0: pIn=A (final Jacobi output), vel=B
                &pressA, &velB,
                // project set 1: pIn=A, vel=A
                &pressA, &velA,
            };
            VkDescriptorSet writeSets[22] = {
                m_advectSets[0], m_advectSets[0], m_advectSets[0], m_advectSets[0],
                m_advectSets[1], m_advectSets[1], m_advectSets[1], m_advectSets[1],
                m_divergenceSets[0], m_divergenceSets[0],
                m_divergenceSets[1], m_divergenceSets[1],
                m_jacobiSets[0], m_jacobiSets[0], m_jacobiSets[0],
                m_jacobiSets[1], m_jacobiSets[1], m_jacobiSets[1],
                m_projectSets[0], m_projectSets[0],
                m_projectSets[1], m_projectSets[1],
            };
            uint32_t writeBindings[22] = {
                0, 1, 2, 3,
                0, 1, 2, 3,
                0, 1,
                0, 1,
                0, 1, 2,
                0, 1, 2,
                0, 1,
                0, 1,
            };

            VkWriteDescriptorSet writes[22] = {};
            for (int32_t i = 0; i < 22; i++)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = writeSets[i];
                writes[i].dstBinding = writeBindings[i];
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = pBufferInfos[i];
            }
            vkUpdateDescriptorSets(device, 22, writes, 0, nullptr);
        }

        VkPipeline FluidKernel3D::CreatePipeline(const char       *pShaderName,
                                                 VkShaderModule   &outShaderModule,
                                                 VkPipelineLayout  pipelineLayout)
        {
            VkDevice device = m_context.Device();

            std::vector<char> shaderCode = Internal::ReadFile(ShaderPath(pShaderName));
            VkShaderModuleCreateInfo shaderInfo = {};
            shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = shaderCode.size();
            shaderInfo.pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            Internal::CheckResult(vkCreateShaderModule(device, &shaderInfo, nullptr, &outShaderModule), "vkCreateShaderModule failed");

            VkPipelineShaderStageCreateInfo stageInfo = {};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = outShaderModule;
            stageInfo.pName = "main";

            VkComputePipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = pipelineLayout;
            VkPipeline pipeline = VK_NULL_HANDLE;
            Internal::CheckResult(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines failed");
            return pipeline;
        }

        void FluidKernel3D::CreatePipelines()
        {
            VkDevice device = m_context.Device();

            VkPushConstantRange range = {};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.offset = 0;

            VkPipelineLayoutCreateInfo layoutInfo = {};
            layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &range;

            range.size = sizeof(FluidAdvectPushConstants);
            layoutInfo.pSetLayouts = &m_advectSetLayout;
            Internal::CheckResult(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_advectPipelineLayout), "vkCreatePipelineLayout failed");

            range.size = sizeof(FluidDivergencePushConstants);
            layoutInfo.pSetLayouts = &m_divergenceSetLayout;
            Internal::CheckResult(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_divergencePipelineLayout), "vkCreatePipelineLayout failed");

            range.size = sizeof(FluidGridPushConstants);
            layoutInfo.pSetLayouts = &m_jacobiSetLayout;
            Internal::CheckResult(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_jacobiPipelineLayout), "vkCreatePipelineLayout failed");

            range.size = sizeof(FluidGridPushConstants);
            layoutInfo.pSetLayouts = &m_projectSetLayout;
            Internal::CheckResult(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_projectPipelineLayout), "vkCreatePipelineLayout failed");

            m_advectPipeline = CreatePipeline("FluidAdvect3D.comp", m_advectShaderModule, m_advectPipelineLayout);
            m_divergencePipeline = CreatePipeline("FluidDivergence3D.comp", m_divergenceShaderModule, m_divergencePipelineLayout);
            m_jacobiPipeline = CreatePipeline("FluidJacobi3D.comp", m_jacobiShaderModule, m_jacobiPipelineLayout);
            m_projectPipeline = CreatePipeline("FluidProject3D.comp", m_projectShaderModule, m_projectPipelineLayout);
        }

        void FluidKernel3D::CreateCommandObjects()
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

        void FluidKernel3D::ZeroInitBuffers()
        {
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            vkCmdFillBuffer(m_commandBuffer, m_velBufferA, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_velBufferB, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_scalBufferA, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_scalBufferB, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_pressureBufferA, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_pressureBufferB, 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(m_commandBuffer, m_divergenceBuffer, 0, VK_WHOLE_SIZE, 0);

            SubmitAndWait();
        }

        void FluidKernel3D::SubmitAndWait()
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

        void FluidKernel3D::RunSteps(std::vector<float> &scalars,
                                     int32_t             steps,
                                     const FluidParams  &params)
        {
            Internal::CheckSize(scalars.size(), static_cast<size_t>(m_length) * 2, "FluidKernel3D::RunSteps scalars");
            VkDeviceSize scalSize = static_cast<VkDeviceSize>(m_length) * 2 * sizeof(float);

            std::memcpy(m_pStagingMapped, scalars.data(), scalSize);

            // The Jacobi ping-pong must end back in pressure buffer A
            // (the project sets read A), so the iteration count is
            // rounded up to even.
            int32_t jacobiIters = params.pressureIterations;
            if ((jacobiIters % 2) != 0)
            {
                jacobiIters++;
            }

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            Internal::CheckResult(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "vkBeginCommandBuffer failed");

            VkBuffer scalCurrent = (m_parity == 0) ? m_scalBufferA : m_scalBufferB;
            VkBufferCopy region = {};
            region.size = scalSize;
            vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, scalCurrent, 1, &region);

            VkMemoryBarrier uploadBarrier = {};
            uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &uploadBarrier, 0, nullptr, 0, nullptr);

            VkMemoryBarrier computeBarrier = {};
            computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

            FluidAdvectPushConstants advectPush = {};
            advectPush.xDim = m_xDim;
            advectPush.yDim = m_yDim;
            advectPush.zDim = m_zDim;
            advectPush.dt = params.dt;
            advectPush.velocityDissipation = params.velocityDissipation;
            advectPush.densityDissipation = params.densityDissipation;
            advectPush.temperatureDissipation = params.temperatureDissipation;
            advectPush.buoyancy = params.buoyancy;
            advectPush.ambientTemperature = params.ambientTemperature;

            FluidDivergencePushConstants divergencePush = {};
            divergencePush.xDim = m_xDim;
            divergencePush.yDim = m_yDim;
            divergencePush.zDim = m_zDim;
            divergencePush.sourceX = params.sourceX;
            divergencePush.sourceY = params.sourceY;
            divergencePush.sourceZ = params.sourceZ;
            divergencePush.sourceRadius = params.sourceRadius;
            divergencePush.sourceStrength = params.sourceStrength;

            FluidGridPushConstants gridPush = {};
            gridPush.xDim = m_xDim;
            gridPush.yDim = m_yDim;
            gridPush.zDim = m_zDim;

            uint32_t groupCountX = (static_cast<uint32_t>(m_xDim) + g_fluidGroupSize - 1) / g_fluidGroupSize;
            uint32_t groupCountY = (static_cast<uint32_t>(m_yDim) + g_fluidGroupSize - 1) / g_fluidGroupSize;
            uint32_t groupCountZ = (static_cast<uint32_t>(m_zDim) + g_fluidGroupSize - 1) / g_fluidGroupSize;

            for (int32_t step = 0; step < steps; step++)
            {
                int32_t parity = (m_parity + step) % 2;

                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_advectPipeline);
                vkCmdPushConstants(m_commandBuffer, m_advectPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(advectPush), &advectPush);
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_advectPipelineLayout, 0, 1, &m_advectSets[parity], 0, nullptr);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_divergencePipeline);
                vkCmdPushConstants(m_commandBuffer, m_divergencePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(divergencePush), &divergencePush);
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_divergencePipelineLayout, 0, 1, &m_divergenceSets[parity], 0, nullptr);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_jacobiPipeline);
                vkCmdPushConstants(m_commandBuffer, m_jacobiPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gridPush), &gridPush);
                for (int32_t iter = 0; iter < jacobiIters; iter++)
                {
                    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_jacobiPipelineLayout, 0, 1, &m_jacobiSets[iter % 2], 0, nullptr);
                    vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
                    vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
                }

                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_projectPipeline);
                vkCmdPushConstants(m_commandBuffer, m_projectPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gridPush), &gridPush);
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_projectPipelineLayout, 0, 1, &m_projectSets[parity], 0, nullptr);
                vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
                vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
            }

            int32_t endParity = (m_parity + steps) % 2;
            VkBuffer scalResult = (endParity == 0) ? m_scalBufferA : m_scalBufferB;

            VkMemoryBarrier readbackBarrier = {};
            readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
            vkCmdCopyBuffer(m_commandBuffer, scalResult, m_stagingBuffer, 1, &region);

            SubmitAndWait();
            m_parity = endParity;

            std::memcpy(scalars.data(), m_pStagingMapped, scalSize);
        }
    }
}
