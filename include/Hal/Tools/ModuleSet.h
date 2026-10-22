#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace Hal
{
    // A model's optional behaviors as a set of modules, each implementing
    // some of a fixed set of hooks (Setup, OnStep, OnDiv, ...). The hooks a
    // module implements are declared when it is added, and Iter(hook)
    // visits only those modules. TModule is the module base type; THook is
    // a caller-defined enum. The set owns its modules; the pointers it
    // hands out stay valid for its lifetime.
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
