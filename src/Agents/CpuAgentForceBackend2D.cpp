#include "Hal/Agents/CpuAgentForceBackend2D.h"

#include <cmath>

namespace Hal
{
    namespace Internal
    {
        // Clamped cell coordinate for an agent position, matching
        // AgentCellCount.comp's CellCoord (out-of-range positions bucket
        // into the edge cell).
        static int32_t AgentForceCellCoord(double  v,
                                           int32_t dim)
        {
            int32_t c = static_cast<int32_t>(std::floor(v));
            if (c < 0)
            {
                c = 0;
            }
            else if (c >= dim)
            {
                c = dim - 1;
            }
            return c;
        }

        // Shortest signed displacement from a to b, wrapping if that is
        // closer — matching GridGeometry2D::DispX/DispY and the Disp
        // helper in AgentForceAccum.comp.
        static double AgentForceDisp(double  a,
                                     double  b,
                                     int32_t dim,
                                     bool    wrap)
        {
            double d = b - a;
            if (wrap)
            {
                double dimF = static_cast<double>(dim);
                if (std::fabs(d) > (dimF * 0.5))
                {
                    d = (a > b) ? (d + dimF) : (d - dimF);
                }
            }
            return d;
        }

        static int32_t AgentForceWrapIndex(int32_t v,
                                           int32_t dim)
        {
            int32_t m = v % dim;
            return (m < 0) ? (m + dim) : m;
        }
    }

    CpuAgentForceBackend2D::CpuAgentForceBackend2D()
        : m_cellOffsets(),
          m_cellCursors(),
          m_bucketAgents()
    {
    }

    CpuAgentForceBackend2D::~CpuAgentForceBackend2D()
    {
    }

    void CpuAgentForceBackend2D::AccumulateForces(const std::vector<double> &positions,
                                                  const std::vector<double> &radii,
                                                  std::vector<double>       &velocityDeltas,
                                                  double                     interactionRad,
                                                  double                     springConstant,
                                                  int32_t                    xDim,
                                                  int32_t                    yDim,
                                                  bool                       wrapX,
                                                  bool                       wrapY)
    {
        int32_t numAgents = static_cast<int32_t>(radii.size());
        int32_t numCells = xDim * yDim;
        velocityDeltas.assign(static_cast<size_t>(numAgents) * 2, 0.0);

        // Pass 1+2: per-cell counts folded straight into exclusive-prefix
        // offsets (cellOffsets[c]..cellOffsets[c+1] is cell c's slot
        // range, exactly the layout the GPU passes build).
        m_cellOffsets.assign(static_cast<size_t>(numCells) + 1, 0);
        for (int32_t i = 0; i < numAgents; i++)
        {
            int32_t cellX = Internal::AgentForceCellCoord(positions[static_cast<size_t>(i) * 2], xDim);
            int32_t cellY = Internal::AgentForceCellCoord(positions[static_cast<size_t>(i) * 2 + 1], yDim);
            m_cellOffsets[static_cast<size_t>(cellX * yDim + cellY) + 1]++;
        }
        for (int32_t c = 0; c < numCells; c++)
        {
            m_cellOffsets[static_cast<size_t>(c) + 1] += m_cellOffsets[static_cast<size_t>(c)];
        }

        // Pass 3: scatter agent indices into their cells' slot ranges.
        m_cellCursors.assign(m_cellOffsets.begin(), m_cellOffsets.end() - 1);
        m_bucketAgents.assign(static_cast<size_t>(numAgents), 0);
        for (int32_t i = 0; i < numAgents; i++)
        {
            int32_t cellX = Internal::AgentForceCellCoord(positions[static_cast<size_t>(i) * 2], xDim);
            int32_t cellY = Internal::AgentForceCellCoord(positions[static_cast<size_t>(i) * 2 + 1], yDim);
            int32_t cell = cellX * yDim + cellY;
            m_bucketAgents[static_cast<size_t>(m_cellCursors[static_cast<size_t>(cell)])] = i;
            m_cellCursors[static_cast<size_t>(cell)]++;
        }

        // Pass 4: per-agent bounding-box-of-cells scan + linear spring
        // sum, the double-precision mirror of AgentForceAccum.comp.
        for (int32_t i = 0; i < numAgents; i++)
        {
            double posX = positions[static_cast<size_t>(i) * 2];
            double posY = positions[static_cast<size_t>(i) * 2 + 1];
            double rad = radii[static_cast<size_t>(i)];

            int32_t xStart = static_cast<int32_t>(std::floor(posX - interactionRad));
            int32_t xEnd = static_cast<int32_t>(std::ceil(posX + interactionRad));
            int32_t yStart = static_cast<int32_t>(std::floor(posY - interactionRad));
            int32_t yEnd = static_cast<int32_t>(std::ceil(posY + interactionRad));

            double velDeltaX = 0.0;
            double velDeltaY = 0.0;
            for (int32_t xSq = xStart; xSq < xEnd; xSq++)
            {
                int32_t retX = xSq;
                bool    inX = (retX >= 0) && (retX < xDim);
                if (!wrapX && !inX)
                {
                    continue;
                }
                if (wrapX && !inX)
                {
                    retX = Internal::AgentForceWrapIndex(retX, xDim);
                }
                for (int32_t ySq = yStart; ySq < yEnd; ySq++)
                {
                    int32_t retY = ySq;
                    bool    inY = (retY >= 0) && (retY < yDim);
                    if (!wrapY && !inY)
                    {
                        continue;
                    }
                    if (wrapY && !inY)
                    {
                        retY = Internal::AgentForceWrapIndex(retY, yDim);
                    }

                    int32_t cell = retX * yDim + retY;
                    int32_t start = m_cellOffsets[static_cast<size_t>(cell)];
                    int32_t end = m_cellOffsets[static_cast<size_t>(cell) + 1];
                    for (int32_t k = start; k < end; k++)
                    {
                        int32_t j = m_bucketAgents[static_cast<size_t>(k)];
                        if (j == i)
                        {
                            continue;
                        }
                        double xComp = Internal::AgentForceDisp(posX, positions[static_cast<size_t>(j) * 2], xDim, wrapX);
                        double yComp = Internal::AgentForceDisp(posY, positions[static_cast<size_t>(j) * 2 + 1], yDim, wrapY);
                        double dist = std::sqrt((xComp * xComp) + (yComp * yComp));
                        if ((dist < interactionRad) && (dist > 0.0))
                        {
                            double touchDist = (rad + radii[static_cast<size_t>(j)]) - dist;
                            double force = springConstant * touchDist;
                            velDeltaX -= (xComp / dist) * force;
                            velDeltaY -= (yComp / dist) * force;
                        }
                    }
                }
            }
            velocityDeltas[static_cast<size_t>(i) * 2] = velDeltaX;
            velocityDeltas[static_cast<size_t>(i) * 2 + 1] = velDeltaY;
        }
    }
}
