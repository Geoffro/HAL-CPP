#pragma once

// The BirthDeath agent model, shared by the BirthDeath example and the
// per-well model in MultiwellExample. Drawing is left to the caller (the
// model only exposes GetAgent / color), keeping it GUI-independent.

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        class BirthDeath;

        class Cell : public Hal::AgentSQ2DUnstackable<BirthDeath>
        {
        public:
            explicit Cell(BirthDeath *pGrid)
                : Hal::AgentSQ2DUnstackable<BirthDeath>(pGrid),
                  m_color(0)
            {
            }

            void Step();

            int32_t Color() const { return m_color; }
            void    SetColor(int32_t color) { m_color = color; }

        private:
            int32_t m_color;
        };

        class BirthDeath : public Hal::AgentGrid2D<Cell, BirthDeath>
        {
        public:
            BirthDeath(int32_t xDim,
                       int32_t yDim,
                       int32_t seedColor)
                : Hal::AgentGrid2D<Cell, BirthDeath>(xDim, yDim),
                  m_mooreHood(Hal::MooreHood(false)),
                  m_color(seedColor)
            {
            }

            void Setup(double rad)
            {
                std::vector<int32_t> coords = Hal::CircleHood(true, rad);
                int32_t nCoords = MapHood(coords.data(), static_cast<int32_t>(coords.size()), Xdim() / 2, Ydim() / 2);
                for (int32_t i = 0; i < nCoords; i++)
                {
                    NewAgentSQ(coords[i]).SetColor(m_color);
                }
            }

            void Step()
            {
                for (Cell &c : *this)
                {
                    c.Step();
                }
                CleanAgents();
                ShuffleAgents(m_rng);
            }

            Hal::Rand &Rng() { return m_rng; }
            int32_t   *MooreHoodData() { return m_mooreHood.data(); }
            int32_t    MooreHoodSize() const { return static_cast<int32_t>(m_mooreHood.size()); }

            static constexpr double DEATH_PROB = 0.01;
            static constexpr double BIRTH_PROB = 0.2;

        private:
            Hal::Rand            m_rng;
            std::vector<int32_t> m_mooreHood;
            int32_t              m_color;
        };

        inline void Cell::Step()
        {
            BirthDeath *pGrid = this->Grid();
            if (pGrid->Rng().Double() < BirthDeath::DEATH_PROB)
            {
                Dispose();
                return;
            }
            if (pGrid->Rng().Double() < BirthDeath::BIRTH_PROB)
            {
                int32_t nOptions = pGrid->MapEmptyHood(pGrid->MooreHoodData(), pGrid->MooreHoodSize(), Xsq(), Ysq());
                if (nOptions > 0)
                {
                    int32_t target = pGrid->MooreHoodData()[pGrid->Rng().Int(nOptions)];
                    pGrid->NewAgentSQ(target).SetColor(m_color);
                }
            }
        }
    }
}
