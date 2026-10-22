#pragma once

#include "Hal/Gpu/Math3D.h"
#include "Hal/Gpu/Window.h"

#include <volk.h>

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Gpu
    {
        // One queued sphere, laid out as SphereLambert.vert's per-instance
        // binding (vec4 center+radius, vec4 color) so the queue uploads as is.
        struct SphereInstance
        {
            float centerX;
            float centerY;
            float centerZ;
            float radius;
            float r;
            float g;
            float b;
            float a;
        };

        // One line-list vertex, laid out as Line3D.vert's binding (vec3
        // position, vec4 color).
        struct LineVertex
        {
            float x;
            float y;
            float z;
            float r;
            float g;
            float b;
            float a;
        };

        // One unit-sphere mesh vertex; the position doubles as the normal.
        struct SphereMeshVertex
        {
            float x;
            float y;
            float z;
        };

        // Depth-tested 3D renderer with the same per-frame usage as the software
        // examples/common/Scene3D: an orbit camera looks at the center of the box
        // [0,xDim]x[0,yDim]x[0,zDim]; each frame call HandleInput() (left-drag
        // orbits, scroll zooms), Clear()/ClearBox(), queue Sphere()/Line()/
        // SphereGlow() primitives, then Render(). Spheres are an instanced
        // unit-sphere mesh with Lambert shading; lines are unlit.
        //
        // Owns its own Vulkan instance and device, with one frame in flight. The
        // swapchain-sized resources are rebuilt on resize or OUT_OF_DATE, and
        // frames are skipped while minimized. Colors are packed ARGB. Geometry
        // buffers double in size when a frame outgrows them (idling the device
        // once).
        class Scene3DRenderContext
        {
        public:
            Scene3DRenderContext(Window &window,
                                 double  xDim,
                                 double  yDim,
                                 double  zDim);
            ~Scene3DRenderContext();

            Scene3DRenderContext(const Scene3DRenderContext &other) = delete;
            Scene3DRenderContext &operator=(const Scene3DRenderContext &other) = delete;

            // Camera orbit in radians (yaw/pitch) and distance from the domain
            // center. Defaults match the software Scene3D.
            void SetCamera(double yaw,
                           double pitch,
                           double distance);

            // Polls the window: left-drag orbits, scroll zooms.
            void HandleInput();

            // Sets the background color the next Render() clears to.
            void Clear(int32_t bgColor);

            // Clear plus the domain bounding-box wireframe.
            void ClearBox(int32_t bgColor,
                          int32_t lineColor);

            // Queues primitives, drawn and consumed by the next Render().
            void Sphere(double  x,
                        double  y,
                        double  z,
                        double  radius,
                        int32_t color);

            // Queues a soft additive glow that fades to nothing at the sphere's
            // edge, drawn after opaque primitives without depth writes, so
            // overlapping glows accumulate in any order (fog, fire). Keep intensity
            // small when many overlap.
            void SphereGlow(double  x,
                            double  y,
                            double  z,
                            double  radius,
                            int32_t color,
                            double  intensity);

            void Line(double  x1,
                      double  y1,
                      double  z1,
                      double  x2,
                      double  y2,
                      double  z2,
                      int32_t color);

            // Draws the queued primitives and presents. Returns false if the frame
            // was skipped (swapchain recreated or window minimized); keep calling.
            // The queue is consumed either way.
            bool Render();

            // Opt-in frame capture: each Render() also copies the presented image
            // to a host-visible buffer for ReadCapturedFrame. Off by default because
            // the readback stalls each frame.
            void SetCaptureEnabled(bool enabled);

            // Copies the most recently presented frame as RGBA8 (row 0 at the top,
            // width*height*4 bytes), waiting for its GPU work. Returns false if
            // capture is disabled.
            bool ReadCapturedFrame(std::vector<uint8_t> &outRgba,
                                   int32_t              &outWidth,
                                   int32_t              &outHeight);

        private:
            void Destroy();

            void CreateInstance();
            void CreateSurface();
            void SelectPhysicalDevice();
            void CreateLogicalDevice();
            void CreateSwapchain();
            void CreateDepthResources();
            void CreateRenderPass();
            void CreateFramebuffers();
            void CreateSphereMesh();
            void CreatePipelines();
            void CreateCommandObjects();
            void CreateSyncObjects();

            VkShaderModule CreateShaderModule(const char *pShaderName);

            // Rebuilds the swapchain, image views, depth image, and framebuffers.
            // Returns false without rebuilding while the framebuffer is zero-sized
            // (minimized).
            bool RecreateSwapchain();

            // Grow the per-frame geometry buffers when a frame queues
            // more than they currently hold (waits for the device to
            // idle, then recreates at least doubled).
            void EnsureSphereCapacity(uint32_t instanceCount);
            void EnsureLineCapacity(uint32_t vertexCount);
            void EnsureGlowCapacity(uint32_t instanceCount);

            // (Re)allocate / free the host-visible capture buffer at the
            // current swapchain extent. No-ops when capture is disabled.
            void CreateCaptureBuffer();
            void DestroyCaptureBuffer();

            // View-projection for the current camera plus the matching
            // headlight direction (unit vector from the domain center
            // toward the camera).
            void ComputeViewProjection(Mat4 &outViewProjection,
                                       Vec3 &outLightDir) const;

            Window &m_window;
            double  m_centerX;
            double  m_centerY;
            double  m_centerZ;
            double  m_maxDim;
            double  m_yaw;
            double  m_pitch;
            double  m_distance;
            float   m_clearColor[4];
            bool    m_dragging;
            double  m_lastMouseX;
            double  m_lastMouseY;

            std::vector<SphereInstance> m_sphereQueue;
            std::vector<LineVertex>     m_lineQueue;
            std::vector<SphereInstance> m_glowQueue;

            VkInstance       m_instance;
            VkSurfaceKHR     m_surface;
            VkPhysicalDevice m_physicalDevice;
            VkDevice         m_device;
            uint32_t         m_graphicsQueueFamilyIndex;
            uint32_t         m_presentQueueFamilyIndex;
            VkQueue          m_graphicsQueue;
            VkQueue          m_presentQueue;

            VkSwapchainKHR           m_swapchain;
            VkFormat                 m_swapchainFormat;
            VkExtent2D               m_swapchainExtent;
            std::vector<VkImage>     m_swapchainImages;
            std::vector<VkImageView> m_swapchainImageViews;

            VkFormat       m_depthFormat;
            VkImage        m_depthImage;
            VkDeviceMemory m_depthMemory;
            VkImageView    m_depthImageView;

            VkRenderPass               m_renderPass;
            std::vector<VkFramebuffer> m_framebuffers;

            VkBuffer       m_meshVertexBuffer;
            VkDeviceMemory m_meshVertexMemory;
            VkBuffer       m_meshIndexBuffer;
            VkDeviceMemory m_meshIndexMemory;
            uint32_t       m_meshIndexCount;

            VkBuffer       m_instanceBuffer;
            VkDeviceMemory m_instanceMemory;
            uint32_t       m_instanceCapacity;
            VkBuffer       m_lineBuffer;
            VkDeviceMemory m_lineMemory;
            uint32_t       m_lineCapacity;
            VkBuffer       m_glowInstanceBuffer;
            VkDeviceMemory m_glowInstanceMemory;
            uint32_t       m_glowInstanceCapacity;

            VkShaderModule   m_sphereVertModule;
            VkShaderModule   m_sphereFragModule;
            VkShaderModule   m_lineVertModule;
            VkShaderModule   m_lineFragModule;
            VkShaderModule   m_glowFragModule;
            VkPipelineLayout m_pipelineLayout;
            VkPipeline       m_spherePipeline;
            VkPipeline       m_linePipeline;
            VkPipeline       m_glowPipeline;

            VkCommandPool                m_commandPool;
            std::vector<VkCommandBuffer> m_commandBuffers;

            VkSemaphore m_imageAvailableSemaphore;
            VkSemaphore m_renderFinishedSemaphore;
            VkFence     m_inFlightFence;

            bool           m_captureEnabled;
            VkBuffer       m_captureBuffer;
            VkDeviceMemory m_captureMemory;
            VkDeviceSize   m_captureSize;
            uint32_t       m_captureWidth;
            uint32_t       m_captureHeight;
        };
    }
}
