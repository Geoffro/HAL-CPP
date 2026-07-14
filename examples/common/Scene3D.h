#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        class GridWindow;

        // A projected screen-space point (structs live at namespace scope,
        // not inside the class, per the coding standard).
        struct ScenePoint
        {
            double sx;     // screen x (pixels)
            double sy;     // screen y (pixels)
            double depth;  // distance from camera (for sorting / radius)
            bool   behind; // true if behind the camera (skip)
        };

        // A queued sphere in world coordinates.
        struct SceneSphere
        {
            double  x;
            double  y;
            double  z;
            double  radius;
            int32_t color;
        };

        // A queued line segment in world coordinates.
        struct SceneLine
        {
            double  x1;
            double  y1;
            double  z1;
            double  x2;
            double  y2;
            double  z2;
            int32_t color;
        };

        // A software 3D scene rendered into a GridWindow's pixel buffer — the
        // analogue of HAL's Gui/OpenGL3DWindow, but CPU-rasterized so it
        // reuses the same pixel-buffer stack as the 2D examples (no Vulkan 3D
        // pipeline; see docs/PHASED_PORT.md Phase 9 for the hardware version).
        //
        // An orbit camera looks at the center of the domain box
        // [0,xDim]x[0,yDim]x[0,zDim]. Queue spheres and line segments each
        // frame, then Render() projects them, depth-sorts (painter's
        // algorithm), and draws them (spheres as depth-scaled discs, lines via
        // Bresenham). On a live window HandleInput() orbits with a left-mouse
        // drag and zooms with the arrow/›‹ keys; headless uses a fixed angle.
        class Scene3D
        {
        public:
            Scene3D(GridWindow &win,
                    int32_t     xDim,
                    int32_t     yDim,
                    int32_t     zDim);

            // Camera controls (radians for yaw/pitch; distance in world units).
            void SetCamera(double yaw,
                           double pitch,
                           double distance);

            // Poll mouse/keyboard on the live window to orbit/zoom the camera.
            // No-op on a headless window.
            void HandleInput();

            void Clear(int32_t bgColor);

            // Clear then draw the domain bounding-box wireframe.
            void ClearBox(int32_t bgColor,
                          int32_t lineColor);

            // Queue primitives (drawn on the next Render()).
            void Sphere(double  x,
                        double  y,
                        double  z,
                        double  radius,
                        int32_t color);

            void Line(double  x1,
                      double  y1,
                      double  z1,
                      double  x2,
                      double  y2,
                      double  z2,
                      int32_t color);

            // Project, depth-sort, rasterize the queued primitives, and present.
            void Render();

        private:
            ScenePoint Project(double x,
                               double y,
                               double z) const;

            void DrawLine3D(double  x1,
                            double  y1,
                            double  z1,
                            double  x2,
                            double  y2,
                            double  z2,
                            int32_t color);

            void DrawPixelLine(int32_t x0,
                               int32_t y0,
                               int32_t x1,
                               int32_t y1,
                               int32_t color);

            GridWindow              &m_win;
            double                   m_cx;
            double                   m_cy;
            double                   m_cz;
            int32_t                  m_maxDim;
            double                   m_yaw;
            double                   m_pitch;
            double                   m_distance;
            double                   m_focal;
            std::vector<SceneSphere> m_spheres;
            std::vector<SceneLine>   m_lines;
            bool                     m_dragging;
            double                   m_lastMouseX;
            double                   m_lastMouseY;
        };
    }
}
