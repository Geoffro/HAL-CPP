#include "Hal/Tools/VarSetManager.h"

#include <stdexcept>

namespace Hal
{
    IVarSet::IVarSet()
    {
    }

    IVarSet::~IVarSet()
    {
    }

    VarSetManager::VarSetManager()
        : m_setup(false),
          m_nParams(0)
    {
    }

    VarSetManager::~VarSetManager()
    {
    }

    int32_t VarSetManager::NewVar()
    {
        if (m_setup)
        {
            throw std::logic_error("VarSetManager::NewVar can't claim parameters after a variable set has been generated");
        }
        m_nParams++;
        return m_nParams - 1;
    }

    int32_t VarSetManager::NumParams() const
    {
        return m_nParams;
    }

    std::vector<double> VarSetManager::GenVarSet()
    {
        m_setup = true;
        return std::vector<double>(static_cast<size_t>(m_nParams), 0.0);
    }

    void VarSetManager::AddVarSet(IVarSet &agent)
    {
        m_setup = true;
        if (agent.GetVars().empty() && (m_nParams > 0))
        {
            agent.SetVars(std::vector<double>(static_cast<size_t>(m_nParams), 0.0));
        }
    }
}
