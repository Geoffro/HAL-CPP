#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace Hal
{
    // Reflection-free replacement for Java's ModuleSetManager (see
    // docs/DESIGN.md — a deliberate redesign, not a faithful port). A model
    // composes optional behavior from a set of modules, each of which
    // implements some subset of a fixed set of hooks (Setup, OnStep, OnDiv,
    // ...). At each hook point the model iterates only the modules that
    // implement that hook.
    //
    // Java discovered which hooks a module implemented by reflecting over
    // overridden methods. C++ has no such reflection, so which hooks a
    // module implements is instead declared explicitly when it is added —
    // the caller, which wrote the module, lists them. TModule is the base
    // module type; THook is a caller-defined enum identifying the hooks
    // (used as a map key, so any enum works).
    //
    // The set owns its modules (added as unique_ptr) and frees them on
    // destruction; the per-hook lists and AddModule's return value are
    // non-owning raw pointers into that storage, stable for the set's
    // lifetime.
    template <typename TModule, typename THook>
    class ModuleSet
    {
    public:
        ModuleSet()
            : m_owned{},
              m_allModules{},
              m_hookModules{},
              m_empty{}
        {
        }

        ~ModuleSet()
        {
        }

        ModuleSet(const ModuleSet &other) = delete;
        ModuleSet &operator=(const ModuleSet &other) = delete;

        // Adds a module, taking ownership, and registers it under each hook
        // it implements. Returns a non-owning pointer to the added module
        // (e.g. to keep a typed handle for direct calls).
        TModule *AddModule(std::unique_ptr<TModule>   module,
                           const std::vector<THook>  &implementedHooks)
        {
            TModule *pModule = module.get();
            m_owned.push_back(std::move(module));
            m_allModules.push_back(pModule);
            for (THook hook : implementedHooks)
            {
                m_hookModules[hook].push_back(pModule);
            }
            return pModule;
        }

        // The modules implementing hook, in the order they were added.
        // Returns an empty list for a hook no module implements.
        const std::vector<TModule *> &Iter(THook hook) const
        {
            auto it = m_hookModules.find(hook);
            if (it == m_hookModules.end())
            {
                return m_empty;
            }
            return it->second;
        }

        // How many modules implement hook.
        int32_t CountModsWithMethod(THook hook) const
        {
            return static_cast<int32_t>(Iter(hook).size());
        }

        // Every module added, in order.
        const std::vector<TModule *> &AllModules() const
        {
            return m_allModules;
        }

    private:
        std::vector<std::unique_ptr<TModule>> m_owned;
        std::vector<TModule *> m_allModules;
        std::map<THook, std::vector<TModule *>> m_hookModules;
        std::vector<TModule *> m_empty;
    };
}
