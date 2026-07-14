#include <doctest/doctest.h>

#include "Hal/Tools/ModuleSet.h"
#include "Hal/Tools/VarSetManager.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------
// VarSetManager
// ---------------------------------------------------------------------

class TestVarSetAgent : public Hal::IVarSet
{
public:
    TestVarSetAgent()
        : m_vars{}
    {
    }

    ~TestVarSetAgent()
    {
    }

    const std::vector<double> &GetVars() const override
    {
        return m_vars;
    }

    void SetVars(std::vector<double> vars) override
    {
        m_vars = std::move(vars);
    }

private:
    std::vector<double> m_vars;
};

TEST_CASE("Hal::VarSetManager: NewVar hands out sequential indices and counts them")
{
    Hal::VarSetManager manager;
    CHECK(manager.NumParams() == 0);
    CHECK(manager.NewVar() == 0);
    CHECK(manager.NewVar() == 1);
    CHECK(manager.NewVar() == 2);
    CHECK(manager.NumParams() == 3);
}

TEST_CASE("Hal::VarSetManager: the layout freezes once a var set is generated")
{
    Hal::VarSetManager manager;
    int32_t a = manager.NewVar();
    int32_t b = manager.NewVar();
    CHECK(a == 0);
    CHECK(b == 1);

    std::vector<double> vars = manager.GenVarSet();
    CHECK(vars.size() == 2);
    CHECK(vars[0] == doctest::Approx(0.0));

    CHECK_THROWS_AS(manager.NewVar(), std::logic_error);
}

TEST_CASE("Hal::VarSetManager: AddVarSet initializes an agent once and leaves an already-set agent alone")
{
    Hal::VarSetManager manager;
    manager.NewVar();
    manager.NewVar();

    TestVarSetAgent agent;
    CHECK(agent.GetVars().empty());

    manager.AddVarSet(agent);
    REQUIRE(agent.GetVars().size() == 2);

    // Mutate, then AddVarSet again — must not clobber existing state.
    agent.SetVars({1.5, 2.5});
    manager.AddVarSet(agent);
    REQUIRE(agent.GetVars().size() == 2);
    CHECK(agent.GetVars()[0] == doctest::Approx(1.5));
    CHECK(agent.GetVars()[1] == doctest::Approx(2.5));
}

TEST_CASE("Hal::VarSetManager: AddVarSet with no claimed parameters leaves the agent empty")
{
    Hal::VarSetManager manager;
    TestVarSetAgent agent;
    manager.AddVarSet(agent);
    CHECK(agent.GetVars().empty());
}

// ---------------------------------------------------------------------
// ModuleSet
// ---------------------------------------------------------------------

enum class TestHook
{
    Setup,
    OnStep,
    OnDivide,
    SetDeathProb
};

class TestModule
{
public:
    TestModule()
        : m_setupCalls(0),
          m_stepCalls(0)
    {
    }

    virtual ~TestModule()
    {
    }

    virtual void Setup()
    {
        m_setupCalls++;
    }

    virtual void OnStep()
    {
        m_stepCalls++;
    }

    virtual double DeathProb()
    {
        return 0.0;
    }

    int32_t m_setupCalls;
    int32_t m_stepCalls;
};

class SetupOnlyModule : public TestModule
{
public:
    SetupOnlyModule()
        : TestModule()
    {
    }

    ~SetupOnlyModule()
    {
    }
};

class DeathModule : public TestModule
{
public:
    explicit DeathModule(double prob)
        : TestModule(),
          m_prob(prob)
    {
    }

    ~DeathModule()
    {
    }

    double DeathProb() override
    {
        return m_prob;
    }

    double m_prob;
};

TEST_CASE("Hal::ModuleSet: Iter returns only the modules registered for each hook")
{
    Hal::ModuleSet<TestModule, TestHook> modules;

    TestModule *pSetupMod = modules.AddModule(std::make_unique<SetupOnlyModule>(), {TestHook::Setup});
    TestModule *pDeathA = modules.AddModule(std::make_unique<DeathModule>(0.1), {TestHook::OnStep, TestHook::SetDeathProb});
    TestModule *pDeathB = modules.AddModule(std::make_unique<DeathModule>(0.2), {TestHook::SetDeathProb});

    CHECK(modules.AllModules().size() == 3);

    // Setup: only the setup module.
    REQUIRE(modules.Iter(TestHook::Setup).size() == 1);
    CHECK(modules.Iter(TestHook::Setup)[0] == pSetupMod);

    // OnStep: only DeathA registered it.
    REQUIRE(modules.Iter(TestHook::OnStep).size() == 1);
    CHECK(modules.Iter(TestHook::OnStep)[0] == pDeathA);

    // SetDeathProb: both death modules, in add order.
    REQUIRE(modules.Iter(TestHook::SetDeathProb).size() == 2);
    CHECK(modules.Iter(TestHook::SetDeathProb)[0] == pDeathA);
    CHECK(modules.Iter(TestHook::SetDeathProb)[1] == pDeathB);

    // A hook no module implements yields an empty list.
    CHECK(modules.Iter(TestHook::OnDivide).empty());
    CHECK(modules.CountModsWithMethod(TestHook::OnDivide) == 0);
    CHECK(modules.CountModsWithMethod(TestHook::SetDeathProb) == 2);
}

TEST_CASE("Hal::ModuleSet: dispatching a hook drives only the registered modules")
{
    Hal::ModuleSet<TestModule, TestHook> modules;
    modules.AddModule(std::make_unique<SetupOnlyModule>(), {TestHook::Setup});
    modules.AddModule(std::make_unique<DeathModule>(0.3), {TestHook::OnStep, TestHook::SetDeathProb});

    for (TestModule *pModule : modules.Iter(TestHook::Setup))
    {
        pModule->Setup();
    }
    for (TestModule *pModule : modules.Iter(TestHook::OnStep))
    {
        pModule->OnStep();
    }

    // Sum death probabilities the same way a model would, over just the
    // modules implementing SetDeathProb.
    double totalDeathProb = 0.0;
    for (TestModule *pModule : modules.Iter(TestHook::SetDeathProb))
    {
        totalDeathProb += pModule->DeathProb();
    }
    CHECK(totalDeathProb == doctest::Approx(0.3));

    // The setup module ran Setup once and never OnStep; the death module
    // ran OnStep once and never Setup.
    CHECK(modules.AllModules()[0]->m_setupCalls == 1);
    CHECK(modules.AllModules()[0]->m_stepCalls == 0);
    CHECK(modules.AllModules()[1]->m_setupCalls == 0);
    CHECK(modules.AllModules()[1]->m_stepCalls == 1);
}
