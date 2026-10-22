#include <doctest/doctest.h>

#include "TestSupport.h"

#include "Hal/Tools/PhylogenyTracker.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// A genome carrying inherited data: a mutation count incremented from
// parent to child.
class TestGenome : public Hal::Genome<TestGenome>
{
public:
    explicit TestGenome(TestGenome *pParent)
        : Hal::Genome<TestGenome>(),
          m_nMutations((pParent != nullptr) ? (pParent->m_nMutations + 1) : 0)
    {
    }

    ~TestGenome()
    {
    }

    int32_t m_nMutations;
};

TEST_CASE("PhylogenyTracker: a fresh progenitor has id 0 and seeds the counts")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);

    CHECK(root.GetId() == 0);
    CHECK(root.GetParent() == nullptr);
    CHECK(root.GetRoot() == &root);
    CHECK(root.GetPop() == 0);
    CHECK(root.m_nMutations == 0);
    CHECK(tracker.GetNumGenomes() == 1);
    CHECK(tracker.GetNumLivingGenomes() == 1);
    CHECK(tracker.GetNumTreeGenomes() == 1);
    CHECK(tracker.GetProgenitor() == &root);
}

TEST_CASE("PhylogenyTracker: inherited data flows parent -> child, ids increase in creation order")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &childA = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&childA);

    CHECK(childA.GetId() == 1);
    CHECK(grandchild.GetId() == 2);
    CHECK(childA.GetParent() == &root);
    CHECK(grandchild.GetParent() == &childA);
    CHECK(childA.m_nMutations == 1);
    CHECK(grandchild.m_nMutations == 2);
    CHECK(grandchild.GetRoot() == &root);
    CHECK(tracker.GetNumGenomes() == 3);
}

TEST_CASE("PhylogenyTracker: population changes track per-genome and phylogeny-wide totals")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);

    root.IncPop();
    root.IncPop();
    child.IncPop();
    CHECK(root.GetPop() == 2);
    CHECK(child.GetPop() == 1);
    CHECK(root.PhylogenyPop() == 3);

    root.DecPop();
    CHECK(root.GetPop() == 1);
    CHECK(root.PhylogenyPop() == 2);

    root.SetPop(5);
    CHECK(root.GetPop() == 5);
    CHECK(root.PhylogenyPop() == 6);
}

TEST_CASE("PhylogenyTracker: SetPop rejects a negative population")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    CHECK_THROWS_AS(root.DecPop(), std::logic_error);
    CHECK_THROWS_AS(root.SetPop(-1), std::logic_error);
}

TEST_CASE("PhylogenyTracker: GetChildren lists direct children newest-first, GetLineage walks to the root")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &childA = tracker.NewGenome(&root);
    TestGenome &childB = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&childB);

    std::vector<TestGenome *> children;
    root.GetChildren(children);
    // firstChild is the most recently added child (B), then A.
    REQUIRE(children.size() == 2);
    CHECK(children[0] == &childB);
    CHECK(children[1] == &childA);

    std::vector<TestGenome *> lineage;
    grandchild.GetLineage(lineage);
    REQUIRE(lineage.size() == 3);
    CHECK(lineage[0] == &grandchild);
    CHECK(lineage[1] == &childB);
    CHECK(lineage[2] == &root);
}

TEST_CASE("PhylogenyTracker: Traverse visits every genome in the subtree exactly once")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &childA = tracker.NewGenome(&root);
    TestGenome &childB = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&childA);

    std::vector<int32_t> visited;
    root.Traverse([&](TestGenome &g)
    {
        visited.push_back(g.GetId());
    });
    std::sort(visited.begin(), visited.end());
    std::vector<int32_t> expected = {root.GetId(), childA.GetId(), childB.GetId(), grandchild.GetId()};
    std::sort(expected.begin(), expected.end());
    CHECK(visited == expected);
}

TEST_CASE("PhylogenyTracker: TraverseWithLineage passes the full root-to-node path at each node")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&child);

    int32_t deepestLineageLength = 0;
    std::vector<TestGenome *> lineage;
    root.TraverseWithLineage(lineage, [&](const std::vector<TestGenome *> &currentLineage)
    {
        // Every reported lineage starts at the root and ends at the
        // visited node.
        CHECK(currentLineage.front() == &root);
        deepestLineageLength = std::max(deepestLineageLength, static_cast<int32_t>(currentLineage.size()));
    });
    CHECK(deepestLineageLength == 3);
    // The traversal must leave the storage as it found it (only the root
    // it seeded remains).
    REQUIRE(lineage.size() == 1);
    CHECK(lineage[0] == &root);
    (void)grandchild;
}

TEST_CASE("PhylogenyTracker: ForEachGenome visits all genomes in id order")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &childA = tracker.NewGenome(&root);
    TestGenome &childB = tracker.NewGenome(&childA);
    (void)childB;

    std::vector<int32_t> ids;
    tracker.ForEachGenome([&](TestGenome &g)
    {
        ids.push_back(g.GetId());
    });
    std::vector<int32_t> expected = {0, 1, 2};
    CHECK(ids == expected);
}

TEST_CASE("PhylogenyTracker: with leaf removal on, a dying childless leaf is pruned from the tree")
{
    Hal::PhylogenyTracker<TestGenome> tracker(true);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();
    TestGenome &child = tracker.NewGenome(&root);
    child.IncPop();

    CHECK(tracker.GetNumTreeGenomes() == 2);
    CHECK(tracker.GetNumLivingGenomes() == 2);

    // Child dies as a childless leaf -> pruned from the tree, and no
    // longer a child of root.
    child.DecPop();
    CHECK(tracker.GetNumLivingGenomes() == 1);
    CHECK(tracker.GetNumTreeGenomes() == 1);

    std::vector<TestGenome *> children;
    root.GetChildren(children);
    CHECK(children.empty());

    // Global-list iteration skips the pruned genome.
    std::vector<int32_t> ids;
    tracker.ForEachGenome([&](TestGenome &g)
    {
        ids.push_back(g.GetId());
    });
    std::vector<int32_t> expected = {0};
    CHECK(ids == expected);
}

TEST_CASE("PhylogenyTracker: leaf removal recursively prunes a dead ancestor branch")
{
    Hal::PhylogenyTracker<TestGenome> tracker(true);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();
    TestGenome &middle = tracker.NewGenome(&root);
    // middle never gains population (stays dead), but has a living child so
    // it can't be pruned yet.
    TestGenome &leaf = tracker.NewGenome(&middle);
    leaf.IncPop();

    CHECK(tracker.GetNumTreeGenomes() == 3);

    // The leaf dies -> it is pruned, and because its parent (middle) is now
    // a childless dead branch, middle is pruned too, up to the still-living
    // root.
    leaf.DecPop();
    CHECK(tracker.GetNumTreeGenomes() == 1);
    std::vector<TestGenome *> children;
    root.GetChildren(children);
    CHECK(children.empty());
}

TEST_CASE("PhylogenyTracker: NewGenome rejects a second progenitor")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    tracker.NewGenome(nullptr);
    CHECK_THROWS_AS(tracker.NewGenome(nullptr), std::logic_error);
}

// ---------------------------------------------------------------------
// Clone recording + CSV export.
// ---------------------------------------------------------------------

static std::vector<std::vector<std::string>> ReadCsv(const std::string &path)
{
    std::vector<std::vector<std::string>> rows;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        std::vector<std::string> fields;
        std::stringstream lineStream(line);
        std::string field;
        while (std::getline(lineStream, field, ','))
        {
            fields.push_back(field);
        }
        rows.push_back(fields);
    }
    return rows;
}

static std::string MutationAttr(const TestGenome &genome)
{
    return std::to_string(genome.m_nMutations);
}

TEST_CASE("PhylogenyTracker::RecordClones stores snapshots and rejects bad labels / leaf removal")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();

    tracker.RecordClones("t0");
    root.IncPop();
    tracker.RecordClones("t1");

    CHECK_THROWS_AS(tracker.RecordClones("has,comma"), std::invalid_argument);
    CHECK_THROWS_AS(tracker.RecordClones("CloneID"), std::invalid_argument);
    CHECK_THROWS_AS(tracker.RecordClones("ParentID"), std::invalid_argument);

    Hal::PhylogenyTracker<TestGenome> leafTracker(true);
    leafTracker.NewGenome(nullptr);
    CHECK_THROWS_AS(leafTracker.RecordClones("t0"), std::logic_error);
}

TEST_CASE("PhylogenyTracker::OutputClonesToCSV with no cutoff writes each clone's own population")
{
    // Linear chain: root(pop 5) -> child(pop 3) -> grandchild(pop 2).
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&child);
    root.SetPop(5);
    child.SetPop(3);
    grandchild.SetPop(2);
    tracker.RecordClones("t0");

    std::string path = TestHelpers::TempFilePath("hal_phylo_nocutoff.csv");
    tracker.OutputClonesToCSV(path, {"nMut"}, MutationAttr, 0);

    std::vector<std::vector<std::string>> rows = ReadCsv(path);
    // Header + 3 clone rows.
    REQUIRE(rows.size() == 4);
    CHECK(rows[0] == std::vector<std::string>{"nMut", "CloneID", "ParentID", "t0"});
    // With no cutoff, the inclusive-then-exclusive transform is the
    // identity, so each clone shows its own recorded population.
    CHECK(rows[1] == std::vector<std::string>{"0", "0", "0", "5"});
    CHECK(rows[2] == std::vector<std::string>{"1", "1", "0", "3"});
    CHECK(rows[3] == std::vector<std::string>{"2", "2", "1", "2"});

    std::remove(path.c_str());
}

TEST_CASE("PhylogenyTracker::OutputClonesToCSV folds a below-cutoff clone into its parent")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);
    TestGenome &grandchild = tracker.NewGenome(&child);
    root.SetPop(5);
    child.SetPop(3);
    grandchild.SetPop(2);
    tracker.RecordClones("t0");

    std::string path = TestHelpers::TempFilePath("hal_phylo_cutoff.csv");
    // Cutoff 3: the grandchild (peak subtree pop 2) is excluded and its
    // population folds into the child, which now shows 5.
    tracker.OutputClonesToCSV(path, {"nMut"}, MutationAttr, 3);

    std::vector<std::vector<std::string>> rows = ReadCsv(path);
    REQUIRE(rows.size() == 3);
    CHECK(rows[0] == std::vector<std::string>{"nMut", "CloneID", "ParentID", "t0"});
    CHECK(rows[1] == std::vector<std::string>{"0", "0", "0", "5"});
    CHECK(rows[2] == std::vector<std::string>{"1", "1", "0", "5"});

    std::remove(path.c_str());
}

TEST_CASE("PhylogenyTracker::OutputClonesToCSV records one column per timepoint")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.SetPop(4);
    tracker.RecordClones("early");
    root.SetPop(9);
    tracker.RecordClones("late");

    std::string path = TestHelpers::TempFilePath("hal_phylo_multi.csv");
    tracker.OutputClonesToCSV(path, {"nMut"}, MutationAttr, 0);

    std::vector<std::vector<std::string>> rows = ReadCsv(path);
    REQUIRE(rows.size() == 2);
    CHECK(rows[0] == std::vector<std::string>{"nMut", "CloneID", "ParentID", "early", "late"});
    CHECK(rows[1] == std::vector<std::string>{"0", "0", "0", "4", "9"});

    std::remove(path.c_str());
}

TEST_CASE("PhylogenyTracker::OutputClonesToCSV validates cutoff, headers, and attribute field counts")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();
    tracker.RecordClones("t0");

    std::string path = TestHelpers::TempFilePath("hal_phylo_validate.csv");
    CHECK_THROWS_AS(tracker.OutputClonesToCSV(path, {"nMut"}, MutationAttr, -1), std::invalid_argument);
    CHECK_THROWS_AS(tracker.OutputClonesToCSV(path, {"CloneID"}, MutationAttr, 0), std::invalid_argument);
    // attrsFn returns one field but two headers are declared.
    CHECK_THROWS_AS(tracker.OutputClonesToCSV(path, {"a", "b"}, MutationAttr, 0), std::invalid_argument);
}

TEST_CASE("PhylogenyTracker: a dead genome stays dead and can't be killed twice")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);
    root.IncPop();
    child.IncPop();
    CHECK(tracker.GetNumLivingGenomes() == 2);

    child.DecPop();
    CHECK_FALSE(child.IsAlive());
    CHECK(tracker.GetNumLivingGenomes() == 1);

    child.SetPop(0);
    CHECK(tracker.GetNumLivingGenomes() == 1);
    CHECK_THROWS_AS(child.IncPop(), std::logic_error);
    CHECK(child.GetPop() == 0);
    CHECK(tracker.GetTotalPop() == 1);
}

TEST_CASE("PhylogenyTracker: with leaf removal, killing an already pruned genome again leaves the tree intact")
{
    Hal::PhylogenyTracker<TestGenome> tracker(true);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();
    TestGenome &a = tracker.NewGenome(&root);
    TestGenome &b = tracker.NewGenome(&root);
    a.IncPop();
    b.IncPop();

    a.DecPop();
    CHECK(tracker.GetNumTreeGenomes() == 2);
    a.SetPop(0);
    CHECK(tracker.GetNumTreeGenomes() == 2);

    std::vector<TestGenome *> children;
    root.GetChildren(children);
    REQUIRE(children.size() == 1);
    CHECK(children[0] == &b);
}

TEST_CASE("PhylogenyTracker::ForEachGenome skips a pruned progenitor")
{
    Hal::PhylogenyTracker<TestGenome> tracker(true);
    TestGenome &root = tracker.NewGenome(nullptr);
    root.IncPop();
    TestGenome &child = tracker.NewGenome(&root);
    child.IncPop();
    TestGenome &grandchild = tracker.NewGenome(&child);
    grandchild.IncPop();

    root.DecPop();
    child.DecPop();
    std::vector<int32_t> ids;
    tracker.ForEachGenome([&](TestGenome &g) { ids.push_back(g.GetId()); });
    CHECK(ids == std::vector<int32_t>{0, 1, 2});

    grandchild.DecPop();
    ids.clear();
    tracker.ForEachGenome([&](TestGenome &g) { ids.push_back(g.GetId()); });
    CHECK(ids.empty());
    CHECK(tracker.GetNumTreeGenomes() == 0);
}

TEST_CASE("PhylogenyTracker::OutputClonesToCSV folds a clone whose peak equals the cutoff into its parent")
{
    Hal::PhylogenyTracker<TestGenome> tracker(false);
    TestGenome &root = tracker.NewGenome(nullptr);
    TestGenome &child = tracker.NewGenome(&root);
    root.SetPop(5);
    child.SetPop(3);
    tracker.RecordClones("t0");

    std::string path = TestHelpers::TempFilePath("hal_phylo_cutoff_equal.csv");
    tracker.OutputClonesToCSV(path, {"nMut"}, MutationAttr, 3);

    std::vector<std::vector<std::string>> rows = ReadCsv(path);
    REQUIRE(rows.size() == 2);
    CHECK(rows[1] == std::vector<std::string>{"0", "0", "0", "8"});

    std::remove(path.c_str());
}
