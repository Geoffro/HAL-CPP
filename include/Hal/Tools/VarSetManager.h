#pragma once

#include <cstdint>
#include <vector>

namespace Hal
{
    // Explicit interface replacing Java's VarSet: an object (typically an
    // agent) that owns a flat array of parameters which one or more modules
    // each claim a slot in (see VarSetManager). Kept an explicit virtual
    // interface — the whole modularity subsystem is being intentionally
    // redesigned away from Java's reflection (see docs/DESIGN.md), and
    // interface initialization is a per-agent-creation event, not a hot
    // path, so the virtual call is immaterial.
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

    // Allocates parameter slots in a shared per-object array, ported from
    // Tools/Modularity/VarSetManager.java. Each module that needs its own
    // per-agent state calls NewVar() once during setup to claim an index;
    // once any variable set has been generated (GenVarSet / AddVarSet), the
    // parameter count is frozen and further NewVar() calls throw.
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
