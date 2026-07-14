#pragma once

// The Outbreak SIR-style model from Examples/MultiwellOutbreak (Person +
// OutbreakWorld), extracted into a header so it can drive both a standalone
// example and the MultiOutbreak multi-well experiment. Chart/GIF drawing is
// dropped; per-cell coloring is exposed via the static ReturnColor so the
// caller (a GridWindow or a MultiWellExperiment drawer) renders it.
//
// Inspired by https://www.meltingasphalt.com/interactive/outbreak/

#include "Hal/Agents/AgentSQ2DUnstackable.h"
#include "Hal/Color.h"
#include "Hal/Grids/AgentGrid2D.h"
#include "Hal/Neighborhood.h"
#include "Hal/Rand.h"

#include <array>
#include <cstdint>
#include <vector>

namespace Hal
{
    namespace Examples
    {
        class OutbreakWorld;

        class Person : public Hal::AgentSQ2DUnstackable<OutbreakWorld>
        {
        public:
            static constexpr int32_t SUSCEPTIBLE = 0;
            static constexpr int32_t DEAD = 1;
            static constexpr int32_t RECOVERED = 2;
            static constexpr int32_t ASYMPTOMATIC_INFECTED = 3;
            static constexpr int32_t SYMPTOMATIC_INFECTED = 4;

            explicit Person(OutbreakWorld *pGrid)
                : Hal::AgentSQ2DUnstackable<OutbreakWorld>(pGrid),
                  m_type(SUSCEPTIBLE),
                  m_futureType(SUSCEPTIBLE),
                  m_quarantined(false),
                  m_dayCounter(0)
            {
            }

            void Init(int32_t t);
            void IncrementCounters();
            void InfectOthers();

            int32_t Type() const { return m_type; }
            void    SetType(int32_t type) { m_type = type; }
            int32_t FutureType() const { return m_futureType; }
            void    SetFutureType(int32_t futureType) { m_futureType = futureType; }
            bool    Quarantined() const { return m_quarantined; }
            void    SetQuarantined(bool quarantined) { m_quarantined = quarantined; }

        private:
            int32_t m_type;
            int32_t m_futureType;
            bool    m_quarantined;
            int32_t m_dayCounter;
        };

        class OutbreakWorld : public Hal::AgentGrid2D<Person, OutbreakWorld>
        {
        public:
            static constexpr int32_t DIMENSION = 100;

            OutbreakWorld()
                : Hal::AgentGrid2D<Person, OutbreakWorld>(DIMENSION, DIMENSION, false, false),
                  m_hood(Hal::RectangleHood(false, TRAVEL_RADIUS, TRAVEL_RADIUS))
            {
                m_counts = {0, 0, 0, 0, 0};
                for (int32_t i = 0; i < Length(); i++)
                {
                    NewAgentSQ(i).Init(Person::SUSCEPTIBLE);
                }
                // Seed a small asymptomatic cluster at the center.
                int32_t c = (Xdim() - 1) / 2;
                GetAgent(c, c)->Init(Person::ASYMPTOMATIC_INFECTED);
                GetAgent(c + 1, c)->Init(Person::ASYMPTOMATIC_INFECTED);
                GetAgent(c - 1, c)->Init(Person::ASYMPTOMATIC_INFECTED);
                GetAgent(c, c + 1)->Init(Person::ASYMPTOMATIC_INFECTED);
                GetAgent(c, c - 1)->Init(Person::ASYMPTOMATIC_INFECTED);
            }

            void StepCells()
            {
                double infectedPercent =
                    (m_counts[Person::ASYMPTOMATIC_INFECTED] + m_counts[Person::SYMPTOMATIC_INFECTED]) /
                    static_cast<double>(DIMENSION * DIMENSION);

                double effectiveFatality = FATALITY_RATE;
                m_aboveCapacity = false;
                if (infectedPercent > HOSPITAL_CAPACITY)
                {
                    effectiveFatality = FATALITY_RATE * FATALITY_MULTIPLIER_OVER_CAPACITY;
                    m_aboveCapacity = true;
                }

                for (Person &c : *this)
                {
                    if (c.Type() == Person::ASYMPTOMATIC_INFECTED)
                    {
                        if (!c.Quarantined())
                        {
                            c.InfectOthers();
                        }
                        if (m_rng.Double() < QUARANTINE_RATE_ASYMPTOMATIC)
                        {
                            c.SetQuarantined(true);
                        }
                        if (m_rng.Double() < effectiveFatality)
                        {
                            c.SetFutureType(Person::DEAD);
                        }
                    }
                    else if (c.Type() == Person::SYMPTOMATIC_INFECTED)
                    {
                        if (!c.Quarantined())
                        {
                            c.InfectOthers();
                        }
                        if (m_rng.Double() < QUARANTINE_RATE_SYMPTOMATIC)
                        {
                            c.SetQuarantined(true);
                        }
                        if (m_rng.Double() < effectiveFatality)
                        {
                            c.SetFutureType(Person::DEAD);
                        }
                    }
                    else if (c.Type() == Person::RECOVERED)
                    {
                        c.SetQuarantined(false);
                    }
                    c.IncrementCounters();
                }

                m_counts = {0, 0, 0, 0, 0};
                for (Person &c : *this)
                {
                    c.SetType(c.FutureType());
                    m_counts[c.Type()]++;
                }
                m_currentTime++;
            }

            static int32_t ReturnColor(int32_t type)
            {
                switch (type)
                {
                    case Person::SUSCEPTIBLE: return Hal::Rgb256(100, 100, 100);
                    case Person::ASYMPTOMATIC_INFECTED: return Hal::Rgb256(255, 221, 226);
                    case Person::SYMPTOMATIC_INFECTED: return Hal::Rgb256(255, 68, 68);
                    case Person::RECOVERED: return Hal::Rgb256(50, 168, 82);
                    case Person::DEAD: return Hal::BLACK;
                    default: return Hal::BLACK;
                }
            }

            std::array<int32_t, 5> &Counts() { return m_counts; }
            bool                    AboveCapacity() const { return m_aboveCapacity; }
            Hal::Rand              &Rng() { return m_rng; }
            int32_t                *HoodData() { return m_hood.data(); }
            int32_t                 HoodSize() const { return static_cast<int32_t>(m_hood.size()); }

            // Mutable model parameters (MultiOutbreak varies quarantine rates
            // per well).
            double  TRANSMISSION = 0.05;
            int32_t ENCOUNTERS_PER_DAY = 8;
            double  QUARANTINE_RATE_SYMPTOMATIC = 0.0;
            double  QUARANTINE_RATE_ASYMPTOMATIC = 0.0;
            double  FATALITY_RATE = 0.01;
            double  HOSPITAL_CAPACITY = 0.005;
            double  FATALITY_MULTIPLIER_OVER_CAPACITY = 2.0;
            int32_t TIME_UNTIL_SYMPTOMATIC = 7;
            int32_t TIME_UNTIL_RECOVERED = 8;

            static constexpr int32_t TRAVEL_RADIUS = 25;

        private:
            std::array<int32_t, 5> m_counts{{0, 0, 0, 0, 0}};
            int32_t                m_currentTime = 0;
            bool                   m_aboveCapacity = false;
            Hal::Rand              m_rng;
            std::vector<int32_t>   m_hood;
        };

        inline void Person::Init(int32_t t)
        {
            m_type = t;
            m_futureType = t;
            m_quarantined = false;
            m_dayCounter = 0;
            this->Grid()->Counts()[t]++;
        }

        inline void Person::IncrementCounters()
        {
            OutbreakWorld *G = this->Grid();
            m_dayCounter++;
            if ((m_type == ASYMPTOMATIC_INFECTED) && (m_dayCounter >= G->TIME_UNTIL_SYMPTOMATIC))
            {
                m_futureType = SYMPTOMATIC_INFECTED;
                m_dayCounter = 0;
            }
            if ((m_type == SYMPTOMATIC_INFECTED) && (m_dayCounter >= G->TIME_UNTIL_RECOVERED))
            {
                m_futureType = RECOVERED;
                m_dayCounter = 0;
            }
        }

        inline void Person::InfectOthers()
        {
            OutbreakWorld *G = this->Grid();
            int32_t        nNeighbors = G->MapOccupiedHood(G->HoodData(), G->HoodSize(), Isq());
            if (nNeighbors == 0)
            {
                return;
            }
            for (int32_t i = 0; i < G->ENCOUNTERS_PER_DAY; i++)
            {
                Person *pNeighbor = G->GetAgent(G->HoodData()[G->Rng().Int(nNeighbors)]);
                if ((pNeighbor->Type() == SUSCEPTIBLE) && !pNeighbor->Quarantined())
                {
                    if (G->Rng().Double() < G->TRANSMISSION)
                    {
                        pNeighbor->SetFutureType(Person::ASYMPTOMATIC_INFECTED);
                    }
                }
            }
        }
    }
}
