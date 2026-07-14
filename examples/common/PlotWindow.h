#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        class GridWindow;

        // A single plotted polyline: a color plus its x/y data points.
        struct PlotSeries
        {
            int32_t             color;
            std::vector<double> xs;
            std::vector<double> ys;
        };

        // A minimal analogue of HAL's Gui/PlotWindow (a UIPlot): a pixel-grid
        // line plotter. Add polylines with AddLine / AddSegment, then Render()
        // maps data coordinates to pixels and rasterizes the segments into an
        // underlying GridWindow (so ToPng and the live window come for free).
        //
        // Axis bounds are either fixed (SetBounds) or, by default, computed
        // to fit all data added so far. Unlike Java's UIPlot this collects
        // points and renders the finished plot rather than streaming live,
        // which yields the same final image with far less machinery.
        class PlotWindow
        {
        public:
            PlotWindow(const std::string &title,
                       int32_t            xPix,
                       int32_t            yPix,
                       int32_t            scaleFactor,
                       bool               active);

            PlotWindow(int32_t xPix,
                       int32_t yPix,
                       int32_t scaleFactor);

            ~PlotWindow();

            PlotWindow(const PlotWindow &other) = delete;
            PlotWindow &operator=(const PlotWindow &other) = delete;

            // Fix the data-space axis bounds. Without this, bounds auto-fit
            // the data (with a small margin).
            void SetBounds(double xMin,
                           double yMin,
                           double xMax,
                           double yMax);

            // Start a new polyline drawn in `color`; subsequent AddSegment
            // calls append to it.
            void AddLine(int32_t color);

            // Append a point to the current line (starting one if needed).
            void AddSegment(double x,
                            double y);

            // Add a whole polyline at once.
            void AddLine(int32_t                    color,
                         const std::vector<double> &xs,
                         const std::vector<double> &ys);

            // Rasterize everything to the pixel buffer (call once data is in).
            void Render(int32_t background);

            bool ToPng(const std::string &path) const;

            // Present the rendered plot; blocks until the window closes (live)
            // or returns immediately (headless).
            void Show();

            GridWindow &Grid();

        private:
            void DataBounds(double &xMin,
                            double &yMin,
                            double &xMax,
                            double &yMax) const;

            void DrawSegment(int32_t x0,
                             int32_t y0,
                             int32_t x1,
                             int32_t y1,
                             int32_t color);

            int32_t                     m_xPix;
            int32_t                     m_yPix;
            std::vector<PlotSeries>     m_lines;
            bool                        m_boundsFixed;
            double                      m_xMin;
            double                      m_yMin;
            double                      m_xMax;
            double                      m_yMax;
            std::unique_ptr<GridWindow> m_grid;
        };
    }
}
