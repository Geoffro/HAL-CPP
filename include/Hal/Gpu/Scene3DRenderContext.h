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
        // One queued sphere, laid out exactly as SphereLambert.vert's
        // per-instance vertex binding expects (vec4 center+radius,
        // vec4 color) so the queue vector can be memcpy'd straight into
        // the instance buffer.
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

        // One line-list vertex, laid out as Line3D.vert's vertex binding
        // expects (vec3 position, vec4 color).
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

        // One unit-sphere mesh vertex. For a unit sphere centered at the
        // origin the normal equals the position, so position is the only
        // attribute the mesh needs.
        struct SphereMeshVertex
        {
            float x;
            float y;
            float z;
        };

        // Hardware Vulkan 3D renderer — the GPU successor to the software
        // examples/common/Scene3D rasterizer, replacing the
        // painter's-algorithm CPU pixel loop with a real depth-tested
        // pipeline. Mirrors Scene3D's per-frame usage
        // shape so models read the same against either renderer: an orbit
        // camera looks at the center of the domain box
        // [0,xDim]x[0,yDim]x[0,zDim]; each frame, HandleInput() (left-drag
        // orbits, scroll zooms — polled through Window, no direct GLFW
        // use), ClearBox()/Clear(), queue Sphere()/Line() primitives, then
        // Render() draws and presents them.
        //
        // What is hardware here, per the Phase 9 scope: a second render
        // pipeline pair with a depth attachment + depth testing (no
        // painter's sort — intersecting spheres resolve per pixel), a real
        // vertex/index-buffered unit-sphere mesh drawn instanced (one
        // instance per queued sphere), line-list geometry for wireframes,
        // a model-view-projection push constant built with Hal::Gpu::Mat4
        // (hand-rolled, no GLM), and Lambert-lit sphere shading
        // (SphereLambert.vert/.frag; Line3D.vert/.frag for unlit lines).
        //
        // Same architectural stance as WindowRenderContext, which is a
        // deliberately parallel class (that one is a fixed 2D
        // textured-quad blitter — 3D needed its own pipelines, render
        // pass, and depth resources, not an extension): owns its own
        // instance/device/swapchain, host-visible/coherent buffers
        // (device-local + staging remains VMA-era polish), and a single
        // frame in flight. Window resizes are handled the same way as
        // the 2D path: the swapchain-sized resources (swapchain, image
        // views, depth image, framebuffers) are rebuilt when the window
        // reports a resize or an acquire/present returns OUT_OF_DATE,
        // frames are skipped while minimized, and the pipelines use
        // dynamic viewport/scissor so they never need recreating.
        //
        // Colors are HAL's packed ARGB int32_t values (see Hal/Color.h).
        // Instance/line buffers grow on demand (the device idles briefly
        // when a frame first exceeds the current capacity, then the
        // doubled buffer is reused thereafter).
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

            // Camera controls (radians for yaw/pitch; distance in world
            // units from the domain center). Defaults match the software
            // Scene3D so the two renderers open on the same view.
            void SetCamera(double yaw,
                           double pitch,
                           double distance);

            // Poll the window's mouse/scroll state: left-drag orbits
            // (yaw/pitch), scroll wheel zooms (distance).
            void HandleInput();

            // Set the background color the next Render() clears to.
            void Clear(int32_t bgColor);

            // Clear plus the domain bounding-box wireframe (12 queued
            // line segments).
            void ClearBox(int32_t bgColor,
                          int32_t lineColor);

            // Queue primitives (drawn and consumed by the next Render()).
            void Sphere(double  x,
                        double  y,
                        double  z,
                        double  radius,
                        int32_t color);

            // Queue a soft additive glow puff: no hard silhouette (the
            // glow fades to nothing at the sphere's edge), drawn with
            // depth writes off and additive blending AFTER the opaque
            // primitives, so overlapping puffs accumulate like fog or
            // fire glow in any order. intensity scales the added light
            // (keep it small when many puffs overlap). This is the
            // volumetric-media answer to Sphere()'s hard surfaces.
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

            // Draw the queued primitives depth-tested and present.
            // Returns true if a frame was presented; false if this frame
            // was skipped because the swapchain had to be recreated
            // (window resize) or the window is minimized — keep calling,
            // the next frame draws normally. The queued primitives are
            // consumed either way.
            bool Render();

            // Frame capture (opt-in). When enabled, each Render() also
            // copies the presented image into a host-visible buffer so
            // ReadCapturedFrame can hand the pixels to a PNG/video encoder.
            // This is what lets an otherwise present-to-screen-only renderer
            // record a clip; it is gated because the readback forces a
            // CPU/GPU sync (a per-frame stall) that a live interactive run
            // should not pay — leave it disabled and Render() is unchanged.
            // Enabling allocates the capture buffer at the current
            // swapchain size (and it is rebuilt across resizes).
            void SetCaptureEnabled(bool enabled);

            // Read the most recently presented frame's pixels as RGBA8
            // (row-major, top-to-bottom, size width*height*4), matching the
            // image on screen. Waits for that frame's GPU work to finish.
            // Returns false if capture is not enabled. Call after a Render()
            // that returned true.
            bool ReadCapturedFrame(std::vector<uint8_t> &outRgba,
                                   int32_t              &outWidth,
                                   int32_t              &outHeight);

        private:
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

            // Rebuilds the swapchain-sized resources (swapchain, image
            // views, depth image, framebuffers) after a resize. Returns
            // false without recreating anything while the framebuffer is
            // zero-sized (minimized) — Render() skips the frame and
            // retries later.
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
