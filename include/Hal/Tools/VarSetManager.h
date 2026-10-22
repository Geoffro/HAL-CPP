#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // An object (typically an agent) owning a flat parameter array in which
    // modules each claim a slot (see VarSetManager).
    class IVarSet
    {
    public:
        IVarSet();
        virtual ~IVarSet();

        // The parameter array. Empty until a VarSetManager has initialized
        // it.
        virtual const std::vector<double> &GetVars() const = 0;
        virtual void SetVars(std::vector<double> vars) = 0;
    };

    // Allocates parameter slots in a shared per-object array. Each module
    // needing per-agent state calls NewVar() once during setup; once any
    // variable set has been generated (GenVarSet / AddVarSet), the layout is
    // frozen and further NewVar() calls throw.
    class VarSetManager
    {
    public:
        VarSetManager();
        ~VarSetManager();

        // Claims and returns the next parameter index. Throws
        // std::logic_error if called after the layout has been frozen.
        int32_t NewVar();

        // The number of parameters claimed so far.
        int32_t NumParams() const;

        // Returns a fresh zero-initialized parameter array of the current
        // size, freezing the layout.
        std::vector<double> GenVarSet();

        // Initializes agent's parameter array (to a fresh zeroed array) if
        // it hasn't been already, freezing the layout. Does nothing if the
        // agent already has one or no parameters were claimed.
        void AddVarSet(IVarSet &agent);

    private:
        bool m_setup;
        int32_t m_nParams;
    };
}
