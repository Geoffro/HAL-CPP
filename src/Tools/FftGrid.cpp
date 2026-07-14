#include "Hal/Tools/FftGrid.h"

#include "Hal/Tools/Internal/FftCore.h"

#include <stdexcept>
#include <utility>

namespace Hal
{
    FftGrid::FftGrid(int32_t xDim,
                     int32_t yDim)
        : m_xDim(xDim),
          m_yDim(yDim),
          m_real(static_cast<size_t>(xDim) * static_cast<size_t>(yDim), 0.0),
          m_imag(static_cast<size_t>(xDim) * static_cast<size_t>(yDim), 0.0)
    {
        if ((Internal::Log2PowerOfTwo(xDim) < 0) || (Internal::Log2PowerOfTwo(yDim) < 0))
        {
            throw std::invalid_argument("Hal::FftGrid requires both dimensions to be powers of two");
        }
    }

    FftGrid::FftGrid(const Grid2D<double> &realGrid)
        : FftGrid(realGrid.Xdim(), realGrid.Ydim())
    {
        SetGrid(realGrid);
    }

    FftGrid::~FftGrid()
    {
    }

    FftGrid::FftGrid(FftGrid &&other) noexcept
        : m_xDim(other.m_xDim),
          m_yDim(other.m_yDim),
          m_real(std::move(other.m_real)),
          m_imag(std::move(other.m_imag))
    {
    }

    FftGrid &FftGrid::operator=(FftGrid &&other) noexcept
    {
        m_xDim = other.m_xDim;
        m_yDim = other.m_yDim;
        m_real = std::move(other.m_real);
        m_imag = std::move(other.m_imag);
        return *this;
    }

    FftGrid FftGrid::Clone() const
    {
        FftGrid copy(m_xDim, m_yDim);
        copy.m_real = m_real;
        copy.m_imag = m_imag;
        return copy;
    }

    int32_t FftGrid::Xdim() const
    {
        return m_xDim;
    }

    int32_t FftGrid::Ydim() const
    {
        return m_yDim;
    }

    int32_t FftGrid::Index(int32_t x,
                           int32_t y) const
    {
        return x * m_yDim + y;
    }

    double FftGrid::GetReal(int32_t x,
                            int32_t y) const
    {
        return m_real[static_cast<size_t>(Index(x, y))];
    }

    double FftGrid::GetImag(int32_t x,
                            int32_t y) const
    {
        return m_imag[static_cast<size_t>(Index(x, y))];
    }

    const std::vector<double> &FftGrid::Real() const
    {
        return m_real;
    }

    const std::vector<double> &FftGrid::Imag() const
    {
        return m_imag;
    }

    void FftGrid::SetGrid(const Grid2D<double> &realGrid)
    {
        if ((realGrid.Xdim() != m_xDim) || (realGrid.Ydim() != m_yDim))
        {
            throw std::invalid_argument("Hal::FftGrid::SetGrid grid dimensions must match");
        }
        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                m_real[static_cast<size_t>(Index(x, y))] = realGrid.Get(x, y);
                m_imag[static_cast<size_t>(Index(x, y))] = 0.0;
            }
        }
    }

    void FftGrid::TransformRows(bool    direct,
                                double  normFactor)
    {
        std::vector<double> rowReal(static_cast<size_t>(m_yDim));
        std::vector<double> rowImag(static_cast<size_t>(m_yDim));
        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                rowReal[static_cast<size_t>(y)] = m_real[static_cast<size_t>(Index(x, y))];
                rowImag[static_cast<size_t>(y)] = m_imag[static_cast<size_t>(Index(x, y))];
            }
            Internal::FftCore(rowReal, rowImag, direct, normFactor);
            for (int32_t y = 0; y < m_yDim; y++)
            {
                m_real[static_cast<size_t>(Index(x, y))] = rowReal[static_cast<size_t>(y)];
                m_imag[static_cast<size_t>(Index(x, y))] = rowImag[static_cast<size_t>(y)];
            }
        }
    }

    void FftGrid::TransformColumns(bool    direct,
                                   double  normFactor)
    {
        std::vector<double> colReal(static_cast<size_t>(m_xDim));
        std::vector<double> colImag(static_cast<size_t>(m_xDim));
        for (int32_t y = 0; y < m_yDim; y++)
        {
            for (int32_t x = 0; x < m_xDim; x++)
            {
                colReal[static_cast<size_t>(x)] = m_real[static_cast<size_t>(Index(x, y))];
                colImag[static_cast<size_t>(x)] = m_imag[static_cast<size_t>(Index(x, y))];
            }
            Internal::FftCore(colReal, colImag, direct, normFactor);
            for (int32_t x = 0; x < m_xDim; x++)
            {
                m_real[static_cast<size_t>(Index(x, y))] = colReal[static_cast<size_t>(x)];
                m_imag[static_cast<size_t>(Index(x, y))] = colImag[static_cast<size_t>(x)];
            }
        }
    }

    void FftGrid::Fft2()
    {
        // Forward: unnormalized (factor 1) on both passes.
        TransformRows(true, 1.0);
        TransformColumns(true, 1.0);
    }

    void FftGrid::Ifft2()
    {
        // Inverse: each pass divides by that pass's length, giving a total
        // 1/(xDim*yDim) factor — matching Java's radice = 1/sqrt(n*n) = 1/n
        // per pass.
        TransformRows(false, 1.0 / static_cast<double>(m_yDim));
        TransformColumns(false, 1.0 / static_cast<double>(m_xDim));
    }

    void FftGrid::FftShift()
    {
        for (int32_t x = 0; x < (m_xDim / 2); x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                int32_t x0 = (x + m_xDim / 2) % m_xDim;
                int32_t y0 = (y + m_yDim / 2) % m_yDim;

                double tempReal = m_real[static_cast<size_t>(Index(x, y))];
                m_real[static_cast<size_t>(Index(x, y))] = m_real[static_cast<size_t>(Index(x0, y0))];
                m_real[static_cast<size_t>(Index(x0, y0))] = tempReal;

                double tempImag = m_imag[static_cast<size_t>(Index(x, y))];
                m_imag[static_cast<size_t>(Index(x, y))] = m_imag[static_cast<size_t>(Index(x0, y0))];
                m_imag[static_cast<size_t>(Index(x0, y0))] = tempImag;
            }
        }
    }

    void FftGrid::ComplexMultiply(const FftGrid &other)
    {
        if ((other.m_xDim != m_xDim) || (other.m_yDim != m_yDim))
        {
            throw std::invalid_argument("Hal::FftGrid::ComplexMultiply grid dimensions must match");
        }
        for (size_t i = 0; i < m_real.size(); i++)
        {
            double r1 = m_real[i];
            double i1 = m_imag[i];
            double r2 = other.m_real[i];
            double i2 = other.m_imag[i];
            m_real[i] = r1 * r2 - i1 * i2;
            m_imag[i] = r1 * i2 + r2 * i1;
        }
    }

    Grid2D<double> FftGrid::ToRealGrid() const
    {
        Grid2D<double> grid(m_xDim, m_yDim);
        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                grid.Set(x, y, m_real[static_cast<size_t>(Index(x, y))]);
            }
        }
        return grid;
    }
}
