#include "Scene3D.h"

#include "GridWindow.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Hal
{
    namespace Examples
    {
        // Sort record for the painter's-algorithm draw order in Render.
        struct SceneDrawKey
        {
            double  depth;
            bool    isSphere;
            int32_t index;
        };

        static bool SceneDrawKeyFartherFirst(const SceneDrawKey &a,
                                             const SceneDrawKey &b)
        {
            return a.depth > b.depth;
        }

        Scene3D::Scene3D(GridWindow &win,
                         int32_t     xDim,
                         int32_t     yDim,
                         int32_t     zDim)
            : m_win(win),
              m_cx(xDim / 2.0),
              m_cy(yDim / 2.0),
              m_cz(zDim / 2.0),
              m_maxDim(std::max(xDim, std::max(yDim, zDim))),
              m_yaw(0.6),
              m_pitch(0.5),
              m_distance(m_maxDim * 2.4),
              m_focal(win.Ydim() * 0.9),
              m_dragging(false),
              m_lastMouseX(0.0),
              m_lastMouseY(0.0)
        {
        }

        void Scene3D::SetCamera(double yaw,
                                double pitch,
                                double distance)
        {
            m_yaw = yaw;
            m_pitch = pitch;
            m_distance = distance;
        }

        void Scene3D::HandleInput()
        {
            void *pHandle = m_win.NativeHandle();
            if (pHandle == nullptr)
            {
                return;
            }
            GLFWwindow *pWindow = static_cast<GLFWwindow *>(pHandle);

            bool   down = glfwGetMouseButton(pWindow, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
            double mx = 0.0;
            double my = 0.0;
            glfwGetCursorPos(pWindow, &mx, &my);
            if (down)
            {
                if (m_dragging)
                {
                    m_yaw += (mx - m_lastMouseX) * 0.01;
                    m_pitch += (my - m_lastMouseY) * 0.01;
                    m_pitch = std::max(-1.5, std::min(1.5, m_pitch));
                }
                m_dragging = true;
                m_lastMouseX = mx;
                m_lastMouseY = my;
            }
            else
            {
                m_dragging = false;
            }

            if (glfwGetKey(pWindow, GLFW_KEY_UP) == GLFW_PRESS)
            {
                m_distance *= 0.98;
            }
            if (glfwGetKey(pWindow, GLFW_KEY_DOWN) == GLFW_PRESS)
            {
                m_distance *= 1.02;
            }
        }

        ScenePoint Scene3D::Project(double x,
                                    double y,
                                    double z) const
        {
            double rx = x - m_cx;
            double ry = y - m_cy;
            double rz = z - m_cz;

            // Yaw about the vertical (y) axis, then pitch about the x axis.
            double cosY = std::cos(m_yaw);
            double sinY = std::sin(m_yaw);
            double x1 = rx * cosY - rz * sinY;
            double z1 = rx * sinY + rz * cosY;
            double y1 = ry;

            double cosP = std::cos(m_pitch);
            double sinP = std::sin(m_pitch);
            double y2 = y1 * cosP - z1 * sinP;
            double z2 = y1 * sinP + z1 * cosP;
            double x2 = x1;

            double     camDepth = z2 + m_distance;
            ScenePoint out;
            out.behind = (camDepth <= 0.1);
            out.depth = camDepth;
            if (out.behind)
            {
                out.sx = 0.0;
                out.sy = 0.0;
                return out;
            }
            // y up on screen; GridWindow's y=0 is the bottom row, so a larger
            // sy is higher up, matching world-up.
            out.sx = m_win.Xdim() / 2.0 + m_focal * x2 / camDepth;
            out.sy = m_win.Ydim() / 2.0 + m_focal * y2 / camDepth;
            return out;
        }

        void Scene3D::Clear(int32_t bgColor)
        {
            m_win.Clear(bgColor);
        }

        void Scene3D::DrawPixelLine(int32_t x0,
                                    int32_t y0,
                                    int32_t x1,
                                    int32_t y1,
                                    int32_t color)
        {
            int32_t dx = std::abs(x1 - x0);
            int32_t dy = -std::abs(y1 - y0);
            int32_t sx = (x0 < x1) ? 1 : -1;
            int32_t sy = (y0 < y1) ? 1 : -1;
            int32_t err = dx + dy;
            while (true)
            {
                m_win.SetPix(x0, y0, color);
                if ((x0 == x1) && (y0 == y1))
                {
                    break;
                }
                int32_t e2 = 2 * err;
                if (e2 >= dy)
                {
                    err += dy;
                    x0 += sx;
                }
                if (e2 <= dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        void Scene3D::DrawLine3D(double  x1,
                                 double  y1,
                                 double  z1,
                                 double  x2,
                                 double  y2,
                                 double  z2,
                                 int32_t color)
        {
            ScenePoint projectedA = Project(x1, y1, z1);
            ScenePoint projectedB = Project(x2, y2, z2);
            if (projectedA.behind || projectedB.behind)
            {
                return;
            }
            DrawPixelLine(static_cast<int32_t>(std::lround(projectedA.sx)), static_cast<int32_t>(std::lround(projectedA.sy)),
                          static_cast<int32_t>(std::lround(projectedB.sx)), static_cast<int32_t>(std::lround(projectedB.sy)), color);
        }

        void Scene3D::ClearBox(int32_t bgColor,
                               int32_t lineColor)
        {
            m_win.Clear(bgColor);

            double x0 = 0.0;
            double y0 = 0.0;
            double z0 = 0.0;
            double x1 = m_cx * 2.0;
            double y1 = m_cy * 2.0;
            double z1 = m_cz * 2.0;

            // 4 edges along x, 4 along y, 4 along z.
            DrawLine3D(x0, y0, z0, x1, y0, z0, lineColor);
            DrawLine3D(x0, y1, z0, x1, y1, z0, lineColor);
            DrawLine3D(x0, y0, z1, x1, y0, z1, lineColor);
            DrawLine3D(x0, y1, z1, x1, y1, z1, lineColor);

            DrawLine3D(x0, y0, z0, x0, y1, z0, lineColor);
            DrawLine3D(x1, y0, z0, x1, y1, z0, lineColor);
            DrawLine3D(x0, y0, z1, x0, y1, z1, lineColor);
            DrawLine3D(x1, y0, z1, x1, y1, z1, lineColor);

            DrawLine3D(x0, y0, z0, x0, y0, z1, lineColor);
            DrawLine3D(x1, y0, z0, x1, y0, z1, lineColor);
            DrawLine3D(x0, y1, z0, x0, y1, z1, lineColor);
            DrawLine3D(x1, y1, z0, x1, y1, z1, lineColor);
        }

        void Scene3D::Sphere(double  x,
                             double  y,
                             double  z,
                             double  radius,
                             int32_t color)
        {
            m_spheres.push_back(SceneSphere{x, y, z, radius, color});
        }

        void Scene3D::Line(double  x1,
                           double  y1,
                           double  z1,
                           double  x2,
                           double  y2,
                           double  z2,
                           int32_t color)
        {
            m_lines.push_back(SceneLine{x1, y1, z1, x2, y2, z2, color});
        }

        void Scene3D::Render()
        {
            // Painter's algorithm: draw farthest primitives first. A draw key
            // records each primitive's sort depth and where to find it. Sphere
            // key is its center depth; line key is its midpoint depth.
            std::vector<SceneDrawKey> order;
            order.reserve(m_spheres.size() + m_lines.size());
            for (int32_t i = 0; i < static_cast<int32_t>(m_spheres.size()); i++)
            {
                const SceneSphere &sphere = m_spheres[static_cast<size_t>(i)];
                order.push_back(SceneDrawKey{Project(sphere.x, sphere.y, sphere.z).depth, true, i});
            }
            for (int32_t i = 0; i < static_cast<int32_t>(m_lines.size()); i++)
            {
                const SceneLine &line = m_lines[static_cast<size_t>(i)];
                double           midDepth = Project((line.x1 + line.x2) / 2, (line.y1 + line.y2) / 2, (line.z1 + line.z2) / 2).depth;
                order.push_back(SceneDrawKey{midDepth, false, i});
            }
            std::sort(order.begin(), order.end(), SceneDrawKeyFartherFirst);

            for (const SceneDrawKey &key : order)
            {
                if (key.isSphere)
                {
                    const SceneSphere &sphere = m_spheres[static_cast<size_t>(key.index)];
                    ScenePoint         point = Project(sphere.x, sphere.y, sphere.z);
                    if (point.behind)
                    {
                        continue;
                    }
                    double screenRadius = sphere.radius * m_focal / point.depth;
                    m_win.Circle(point.sx, point.sy, screenRadius, sphere.color);
                }
                else
                {
                    const SceneLine &line = m_lines[static_cast<size_t>(key.index)];
                    DrawLine3D(line.x1, line.y1, line.z1, line.x2, line.y2, line.z2, line.color);
                }
            }

            m_spheres.clear();
            m_lines.clear();
            m_win.Tick();
        }
    }
}
