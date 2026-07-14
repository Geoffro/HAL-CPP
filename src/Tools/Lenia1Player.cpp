#include "Hal/Tools/Lenia1Player.h"

#include "Hal/MathUtil.h"

#include <utility>

namespace Hal
{
    Lenia1Player::Lenia1Player(int32_t                        sideLenExp,
                               double                         deltaT,
                               std::function<double(double)>  kernelFn,
                               std::function<double(double)>  growthFn)
        : m_xDim(1 << sideLenExp),
          m_yDim(1 << sideLenExp),
          m_length(m_xDim * m_yDim),
          m_deltaT(deltaT),
          m_tick(0),
          m_kSum(0.0),
          m_kernelFn(std::move(kernelFn)),
          m_growthFn(std::move(growthFn)),
          m_field(m_xDim, m_yDim, true, true),
          m_kernelField(m_xDim, m_yDim),
          m_fftField(m_xDim, m_yDim),
          m_fftKernel(m_xDim, m_yDim)
    {
        RecalcKernel();
    }

    Lenia1Player::~Lenia1Player()
    {
    }

    void Lenia1Player::SetKernelFunction(std::function<double(double)> kernelFn)
    {
        m_kernelFn = std::move(kernelFn);
        RecalcKernel();
    }

    void Lenia1Player::SetGrowthFunction(std::function<double(double)> growthFn)
    {
        m_growthFn = std::move(growthFn);
    }

    void Lenia1Player::RecalcKernel()
    {
        m_kSum = 0.0;
        for (int32_t x = 0; x < m_xDim; x++)
        {
            for (int32_t y = 0; y < m_yDim; y++)
            {
                double rxy = Dist(static_cast<double>(x), static_cast<double>(y), m_xDim / 2.0, m_yDim / 2.0);
                double val = m_kernelFn(rxy);
                m_kSum += val;
                m_kernelField.Set(x, y, val);
            }
        }

        // Normalize the kernel to sum to 1 (so convolving a constant field
        // leaves it unchanged), then shift its center to the origin and
        // transform — the FFT of the kernel never changes between steps.
        for (int32_t i = 0; i < m_length; i++)
        {
            m_kernelField.Set(i, m_kernelField.Get(i) / m_kSum);
        }
        m_fftKernel.SetGrid(m_kernelField);
        m_fftKernel.FftShift();
        m_fftKernel.Fft2();
    }

    void Lenia1Player::Update()
    {
        m_fftField.SetGrid(m_field);
        m_fftField.Fft2();
        m_fftField.ComplexMultiply(m_fftKernel);
        m_fftField.Ifft2();

        for (int32_t i = 0; i < m_length; i++)
        {
            double convolved = m_fftField.Real()[static_cast<size_t>(i)];
            double newVal = Bound(m_field.Get(i) + m_growthFn(convolved) * m_deltaT, 0.0, 1.0);
            m_field.Set(i, newVal);
        }
        m_tick++;
    }

    int32_t Lenia1Player::Xdim() const
    {
        return m_xDim;
    }

    int32_t Lenia1Player::Ydim() const
    {
        return m_yDim;
    }

    int32_t Lenia1Player::Length() const
    {
        return m_length;
    }

    int32_t Lenia1Player::GetTick() const
    {
        return m_tick;
    }

    void Lenia1Player::ResetTick()
    {
        m_tick = 0;
    }

    double Lenia1Player::Get(int32_t i) const
    {
        return m_field.Get(i);
    }

    double Lenia1Player::Get(int32_t x,
                             int32_t y) const
    {
        return m_field.Get(x, y);
    }

    void Lenia1Player::Set(int32_t i,
                           double  val)
    {
        m_field.Set(i, val);
    }

    void Lenia1Player::Set(int32_t x,
                           int32_t y,
                           double  val)
    {
        m_field.Set(x, y, val);
    }

    const Grid2D<double> &Lenia1Player::GetField() const
    {
        return m_field;
    }

    double Lenia1Player::GetConvolvedFieldVal(int32_t i) const
    {
        return m_fftField.Real()[static_cast<size_t>(i)];
    }

    double Lenia1Player::GetConvolvedFieldVal(int32_t x,
                                              int32_t y) const
    {
        return m_fftField.GetReal(x, y);
    }

    double Lenia1Player::GetKernelVal(int32_t i) const
    {
        return m_kernelField.Get(i) * m_kSum;
    }

    double Lenia1Player::GetKernelVal(int32_t x,
                                      int32_t y) const
    {
        return m_kernelField.Get(x, y) * m_kSum;
    }
}
