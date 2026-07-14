#include "Hal/Tools/LeniaNPlayer.h"

#include "Hal/Log/Logger.h"
#include "Hal/MathUtil.h"

#include <stdexcept>
#include <utility>

namespace Hal
{
    LeniaNPlayer::LeniaNPlayer(int32_t  sideLenExp,
                               int32_t  nPlayers,
                               double   deltaT,
                               KernelFn kernelFn,
                               GrowthFn growthFn)
        : m_nPlayers(nPlayers),
          m_deltaT(deltaT),
          m_tick(0),
          m_xDim(1 << sideLenExp),
          m_yDim(1 << sideLenExp),
          m_length(m_xDim * m_yDim),
          m_kernelFn(std::move(kernelFn)),
          m_growthFn(std::move(growthFn)),
          m_fields{},
          m_kernelFields{},
          m_kSums(static_cast<size_t>(nPlayers) * static_cast<size_t>(nPlayers), 0.0),
          m_fftKernels{},
          m_fftWork(m_xDim, m_yDim),
          m_convolved(static_cast<size_t>(nPlayers) * static_cast<size_t>(nPlayers) * static_cast<size_t>(m_length), 0.0),
          m_growthScratch(static_cast<size_t>(nPlayers), 0.0)
    {
        m_fields.reserve(static_cast<size_t>(nPlayers));
        for (int32_t i = 0; i < nPlayers; i++)
        {
            m_fields.emplace_back(m_xDim, m_yDim, true, true);
        }
        m_kernelFields.reserve(static_cast<size_t>(nPlayers) * static_cast<size_t>(nPlayers));
        m_fftKernels.reserve(static_cast<size_t>(nPlayers) * static_cast<size_t>(nPlayers));
        for (int32_t i = 0; i < nPlayers; i++)
        {
            for (int32_t j = 0; j < nPlayers; j++)
            {
                m_kernelFields.emplace_back(m_xDim, m_yDim);
                m_fftKernels.emplace_back(m_xDim, m_yDim);
            }
        }
        RecalcKernels();
    }

    LeniaNPlayer::~LeniaNPlayer()
    {
    }

    void LeniaNPlayer::SetKernelFunction(KernelFn kernelFn)
    {
        m_kernelFn = std::move(kernelFn);
        RecalcKernels();
    }

    void LeniaNPlayer::SetGrowthFunction(GrowthFn growthFn)
    {
        m_growthFn = std::move(growthFn);
    }

    void LeniaNPlayer::StoreKernel(const std::string &label)
    {
        LeniaKernelSnapshot snapshot;
        snapshot.kernelFn = m_kernelFn;
        snapshot.kernelFields = m_kernelFields;
        snapshot.kSums = m_kSums;
        snapshot.fftKernels.reserve(m_fftKernels.size());
        for (const FftGrid &kernel : m_fftKernels)
        {
            snapshot.fftKernels.push_back(kernel.Clone());
        }
        m_kernelStorage[label] = std::move(snapshot);
    }

    void LeniaNPlayer::LoadKernel(const std::string &label)
    {
        std::map<std::string, LeniaKernelSnapshot>::const_iterator it = m_kernelStorage.find(label);
        if (it == m_kernelStorage.end())
        {
            Logger::Instance().Log(LogSeverity::Error, "LeniaNPlayer::LoadKernel: no stored kernel labeled \"" + label + "\"");
            throw std::invalid_argument("LeniaNPlayer::LoadKernel: unknown kernel label");
        }
        const LeniaKernelSnapshot &snapshot = it->second;
        m_kernelFn = snapshot.kernelFn;
        m_kernelFields = snapshot.kernelFields;
        m_kSums = snapshot.kSums;
        m_fftKernels.clear();
        m_fftKernels.reserve(snapshot.fftKernels.size());
        for (const FftGrid &kernel : snapshot.fftKernels)
        {
            m_fftKernels.push_back(kernel.Clone());
        }
    }

    int32_t LeniaNPlayer::KernelIndex(int32_t iPlayer,
                                      int32_t jPlayer) const
    {
        return iPlayer * m_nPlayers + jPlayer;
    }

    int32_t LeniaNPlayer::ConvolvedIndex(int32_t iPlayer,
                                         int32_t jPlayer,
                                         int32_t cell) const
    {
        return (iPlayer * m_nPlayers + jPlayer) * m_length + cell;
    }

    void LeniaNPlayer::RecalcKernels()
    {
        for (int32_t i = 0; i < m_nPlayers; i++)
        {
            for (int32_t j = 0; j < m_nPlayers; j++)
            {
                int32_t idx = KernelIndex(i, j);
                Grid2D<double> &kernel = m_kernelFields[static_cast<size_t>(idx)];
                double kSum = 0.0;
                for (int32_t x = 0; x < m_xDim; x++)
                {
                    for (int32_t y = 0; y < m_yDim; y++)
                    {
                        double rxy = Dist(static_cast<double>(x), static_cast<double>(y), m_xDim / 2.0, m_yDim / 2.0);
                        double val = m_kernelFn(i, j, rxy);
                        kSum += val;
                        kernel.Set(x, y, val);
                    }
                }
                m_kSums[static_cast<size_t>(idx)] = kSum;
                for (int32_t k = 0; k < m_length; k++)
                {
                    kernel.Set(k, kernel.Get(k) / kSum);
                }
                m_fftKernels[static_cast<size_t>(idx)].SetGrid(kernel);
                m_fftKernels[static_cast<size_t>(idx)].FftShift();
                m_fftKernels[static_cast<size_t>(idx)].Fft2();
            }
        }
    }

    void LeniaNPlayer::Update()
    {
        // Convolve every source field A_j with every kernel K_ij, storing
        // the real part of each K_ij*A_j.
        for (int32_t i = 0; i < m_nPlayers; i++)
        {
            for (int32_t j = 0; j < m_nPlayers; j++)
            {
                m_fftWork.SetGrid(m_fields[static_cast<size_t>(j)]);
                m_fftWork.Fft2();
                m_fftWork.ComplexMultiply(m_fftKernels[static_cast<size_t>(KernelIndex(i, j))]);
                m_fftWork.Ifft2();
                const std::vector<double> &real = m_fftWork.Real();
                int32_t base = ConvolvedIndex(i, j, 0);
                for (int32_t k = 0; k < m_length; k++)
                {
                    m_convolved[static_cast<size_t>(base + k)] = real[static_cast<size_t>(k)];
                }
            }
        }

        // Update each channel from its growth function over all its
        // convolutions.
        for (int32_t i = 0; i < m_nPlayers; i++)
        {
            Grid2D<double> &field = m_fields[static_cast<size_t>(i)];
            for (int32_t k = 0; k < m_length; k++)
            {
                for (int32_t j = 0; j < m_nPlayers; j++)
                {
                    m_growthScratch[static_cast<size_t>(j)] = m_convolved[static_cast<size_t>(ConvolvedIndex(i, j, k))];
                }
                double newVal = Bound(field.Get(k) + m_growthFn(i, m_growthScratch) * m_deltaT, 0.0, 1.0);
                field.Set(k, newVal);
            }
        }
        m_tick++;
    }

    int32_t LeniaNPlayer::NPlayers() const
    {
        return m_nPlayers;
    }

    int32_t LeniaNPlayer::Xdim() const
    {
        return m_xDim;
    }

    int32_t LeniaNPlayer::Ydim() const
    {
        return m_yDim;
    }

    int32_t LeniaNPlayer::Length() const
    {
        return m_length;
    }

    int32_t LeniaNPlayer::GetTick() const
    {
        return m_tick;
    }

    void LeniaNPlayer::ResetTick()
    {
        m_tick = 0;
    }

    double LeniaNPlayer::Get(int32_t iPlayer,
                             int32_t i) const
    {
        return m_fields[static_cast<size_t>(iPlayer)].Get(i);
    }

    double LeniaNPlayer::Get(int32_t iPlayer,
                             int32_t x,
                             int32_t y) const
    {
        return m_fields[static_cast<size_t>(iPlayer)].Get(x, y);
    }

    void LeniaNPlayer::Set(int32_t iPlayer,
                           int32_t i,
                           double  val)
    {
        m_fields[static_cast<size_t>(iPlayer)].Set(i, val);
    }

    void LeniaNPlayer::Set(int32_t iPlayer,
                           int32_t x,
                           int32_t y,
                           double  val)
    {
        m_fields[static_cast<size_t>(iPlayer)].Set(x, y, val);
    }

    const Grid2D<double> &LeniaNPlayer::GetField(int32_t iPlayer) const
    {
        return m_fields[static_cast<size_t>(iPlayer)];
    }

    double LeniaNPlayer::GetConvolvedFieldVal(int32_t iPlayer,
                                              int32_t jPlayer,
                                              int32_t i) const
    {
        return m_convolved[static_cast<size_t>(ConvolvedIndex(iPlayer, jPlayer, i))];
    }

    double LeniaNPlayer::GetConvolvedFieldVal(int32_t iPlayer,
                                              int32_t jPlayer,
                                              int32_t x,
                                              int32_t y) const
    {
        return m_convolved[static_cast<size_t>(ConvolvedIndex(iPlayer, jPlayer, x * m_yDim + y))];
    }

    double LeniaNPlayer::GetKernelVal(int32_t iPlayer,
                                      int32_t jPlayer,
                                      int32_t i) const
    {
        int32_t idx = KernelIndex(iPlayer, jPlayer);
        return m_kernelFields[static_cast<size_t>(idx)].Get(i) * m_kSums[static_cast<size_t>(idx)];
    }

    double LeniaNPlayer::GetKernelVal(int32_t iPlayer,
                                      int32_t jPlayer,
                                      int32_t x,
                                      int32_t y) const
    {
        int32_t idx = KernelIndex(iPlayer, jPlayer);
        return m_kernelFields[static_cast<size_t>(idx)].Get(x, y) * m_kSums[static_cast<size_t>(idx)];
    }
}
