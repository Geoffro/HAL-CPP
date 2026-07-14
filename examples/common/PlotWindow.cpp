#include "PlotWindow.h"

#include "GridWindow.h"

#include "Hal/MathUtil.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Hal
{
    namespace Examples
    {
        PlotWindow::PlotWindow(const std::string &title,
                               int32_t            xPix,
                               int32_t            yPix,
                               int32_t            scaleFactor,
                               bool               active)
            : m_xPix(xPix),
              m_yPix(yPix),
              m_boundsFixed(false),
              m_xMin(0.0),
              m_yMin(0.0),
              m_xMax(1.0),
              m_yMax(1.0),
              m_grid(std::make_unique<GridWindow>(title, xPix, yPix, scaleFactor, active))
        {
        }

        PlotWindow::PlotWindow(int32_t xPix,
                               int32_t yPix,
                               int32_t scaleFactor)
            : PlotWindow("HAL Plot", xPix, yPix, scaleFactor, true)
        {
        }

        PlotWindow::~PlotWindow()
        {
        }

        void PlotWindow::SetBounds(double xMin,
                                   double yMin,
                                   double xMax,
                                   double yMax)
        {
            m_boundsFixed = true;
            m_xMin = xMin;
            m_yMin = yMin;
            m_xMax = xMax;
            m_yMax = yMax;
        }

        void PlotWindow::AddLine(int32_t color)
        {
            m_lines.push_back(PlotSeries{color, {}, {}});
        }

        void PlotWindow::AddSegment(double x,
                                    double y)
        {
            if (m_lines.empty())
            {
                AddLine(0);
            }
            m_lines.back().xs.push_back(x);
            m_lines.back().ys.push_back(y);
        }

        void PlotWindow::AddLine(int32_t                    color,
                                 const std::vector<double> &xs,
                                 const std::vector<double> &ys)
        {
            m_lines.push_back(PlotSeries{color, xs, ys});
        }

        void PlotWindow::DataBounds(double &xMin,
                                    double &yMin,
                                    double &xMax,
                                    double &yMax) const
        {
            if (m_boundsFixed)
            {
                xMin = m_xMin;
                yMin = m_yMin;
                xMax = m_xMax;
                yMax = m_yMax;
                return;
            }
            xMin = std::numeric_limits<double>::max();
            yMin = std::numeric_limits<double>::max();
            xMax = std::numeric_limits<double>::lowest();
            yMax = std::numeric_limits<double>::lowest();
            for (const PlotSeries &line : m_lines)
            {
                for (double x : line.xs)
                {
                    xMin = std::min(xMin, x);
                    xMax = std::max(xMax, x);
                }
                for (double y : line.ys)
                {
                    yMin = std::min(yMin, y);
                    yMax = std::max(yMax, y);
                }
            }
            if (xMax <= xMin)
            {
                xMin -= 0.5;
                xMax += 0.5;
            }
            if (yMax <= yMin)
            {
                yMin -= 0.5;
                yMax += 0.5;
            }
            // A small margin so extreme points aren't glued to the edge.
            double yMargin = (yMax - yMin) * 0.05;
            yMin -= yMargin;
            yMax += yMargin;
        }

        void PlotWindow::DrawSegment(int32_t x0,
                                     int32_t y0,
                                     int32_t x1,
                                     int32_t y1,
                                     int32_t color)
        {
            // Integer Bresenham line into the pixel buffer.
            int32_t dx = std::abs(x1 - x0);
            int32_t dy = -std::abs(y1 - y0);
            int32_t sx = (x0 < x1) ? 1 : -1;
            int32_t sy = (y0 < y1) ? 1 : -1;
            int32_t err = dx + dy;
            while (true)
            {
                m_grid->SetPix(x0, y0, color);
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

        void PlotWindow::Render(int32_t background)
        {
            m_grid->Clear(background);

            double xMin;
            double yMin;
            double xMax;
            double yMax;
            DataBounds(xMin, yMin, xMax, yMax);

            double xSpan = (xMax - xMin);
            double ySpan = (yMax - yMin);
            if (xSpan == 0.0)
            {
                xSpan = 1.0;
            }
            if (ySpan == 0.0)
            {
                ySpan = 1.0;
            }

            auto toPixX = [&](double x) {
                return static_cast<int32_t>(std::lround((x - xMin) / xSpan * (m_xPix - 1)));
            };
            auto toPixY = [&](double y) {
                return static_cast<int32_t>(std::lround((y - yMin) / ySpan * (m_yPix - 1)));
            };

            for (const PlotSeries &line : m_lines)
            {
                for (size_t i = 1; i < line.xs.size(); i++)
                {
                    DrawSegment(toPixX(line.xs[i - 1]), toPixY(line.ys[i - 1]),
                                toPixX(line.xs[i]), toPixY(line.ys[i]), line.color);
                }
            }
            // Clear the queued lines so each Render is an independent batch
            // (callers that re-plot every frame rebuild them via AddLine).
            m_lines.clear();
            m_grid->Tick();
        }

        bool PlotWindow::ToPng(const std::string &path) const
        {
            return m_grid->ToPng(path);
        }

        void PlotWindow::Show()
        {
            while (!m_grid->IsClosed())
            {
                m_grid->TickPause(50);
            }
        }

        GridWindow &PlotWindow::Grid()
        {
            return *m_grid;
        }
    }
}
