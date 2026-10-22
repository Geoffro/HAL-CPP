#pragma once

#include "Hal/Log/Logger.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Hal
{
    // Phylogeny tracking. A Genome is one heritable state (e.g. a clone's
    // mutation set), created once per mutation and shared by pointer among
    // the agents carrying it; the parent/child tree plus per-genome living
    // populations is the phylogeny. TDerived is the concrete genome class
    // (CRTP), e.g.
    //     class CellGenome : public Hal::Genome<CellGenome> { ... };
    // The tracker owns every genome (created with NewGenome) in a
    // std::deque, so genome pointers stay valid for its lifetime. A genome
    // whose population reaches zero is dead; with leaf removal on, dead
    // childless genomes (and any resulting dead, childless ancestors) are
    // pruned from the tree.
    template <typename TDerived>
    class PhylogenyTracker;

    template <typename TDerived>
    class Genome
    {
    public:
        Genome()
            : m_pop(0),
              m_id(-1),
              m_pParent(nullptr),
              m_pFirstChild(nullptr),
              m_pNextSibling(nullptr),
              m_pPrevSibling(nullptr),
              m_pNext(nullptr),
              m_pPrev(nullptr),
              m_pTracker(nullptr),
              m_alive(false)
        {
        }

        ~Genome()
        {
        }

        Genome(const Genome &other) = delete;
        Genome &operator=(const Genome &other) = delete;

        // Id in creation order (0 for the progenitor), stable for the
        // genome's lifetime.
        int32_t GetId() const
        {
            return m_id;
        }

        // The living population currently carrying this genome.
        int64_t GetPop() const
        {
            return m_pop;
        }

        TDerived *GetParent() const
        {
            return m_pParent;
        }

        // The progenitor (root) of the whole phylogeny.
        TDerived *GetRoot() const
        {
            return m_pTracker->m_pProgenitor;
        }

        // Sets this genome's living population and the phylogeny-wide
        // total. Reaching zero kills the genome (see the class comment).
        // Throws std::logic_error for a negative population or a positive
        // one on a dead genome.
        void SetPop(int64_t pop)
        {
            if (m_pTracker == nullptr)
            {
                throw std::logic_error("Genome must be part of a tracker before its pop can be changed");
            }
            if (pop < 0)
            {
                throw std::logic_error("Genome pop can't be decreased below 0");
            }
            if (!m_alive && (pop > 0))
            {
                throw std::logic_error("Genome is dead; its pop can't be increased");
            }
            m_pTracker->m_totalPop += pop - m_pop;
            m_pop = pop;
            if ((pop == 0) && m_alive)
            {
                KillGenome();
            }
        }

        bool IsAlive() const
        {
            return m_alive;
        }

        void IncPop()
        {
            SetPop(m_pop + 1);
        }

        void DecPop()
        {
            SetPop(m_pop - 1);
        }

        int32_t GetNumGenomes() const
        {
            return m_pTracker->m_nGenomesEver;
        }

        int32_t GetNumLivingGenomes() const
        {
            return m_pTracker->m_nLivingGenomes;
        }

        int32_t GetNumTreeGenomes() const
        {
            return m_pTracker->m_nTreeGenomes;
        }

        // Total living population across the whole phylogeny.
        int64_t PhylogenyPop() const
        {
            return m_pTracker->m_totalPop;
        }

        // Appends every direct child of this genome to childrenStorage.
        void GetChildren(std::vector<TDerived *> &childrenStorage) const
        {
            TDerived *pChild = m_pFirstChild;
            while (pChild != nullptr)
            {
                childrenStorage.push_back(pChild);
                pChild = pChild->m_pNextSibling;
            }
        }

        // Appends this genome and all its ancestors up to the root to
        // lineageStorage (nearest ancestor first, root last).
        void GetLineage(std::vector<TDerived *> &lineageStorage)
        {
            TDerived *pAncestor = Self();
            while (pAncestor != nullptr)
            {
                lineageStorage.push_back(pAncestor);
                pAncestor = pAncestor->m_pParent;
            }
        }

        // Runs fn(TDerived&) on this genome and every descendant
        // (depth-first).
        template <typename GenomeFn>
        void Traverse(GenomeFn fn)
        {
            fn(*Self());
            TDerived *pChild = m_pFirstChild;
            while (pChild != nullptr)
            {
                pChild->Traverse(fn);
                pChild = pChild->m_pNextSibling;
            }
        }

        // Depth-first traversal that passes fn the lineage from this genome
        // down to each visited descendant (as the growing lineageStorage
        // vector). fn signature: fn(const std::vector<TDerived*>&).
        template <typename LineageFn>
        void TraverseWithLineage(std::vector<TDerived *> &lineageStorage,
                                 LineageFn                fn)
        {
            lineageStorage.push_back(Self());
            fn(lineageStorage);
            TDerived *pChild = m_pFirstChild;
            while (pChild != nullptr)
            {
                pChild->TraverseWithLineage(lineageStorage, fn);
                lineageStorage.pop_back();
                pChild = pChild->m_pNextSibling;
            }
        }

    private:
        friend class PhylogenyTracker<TDerived>;

        TDerived *Self()
        {
            return static_cast<TDerived *>(this);
        }

        // Links a freshly created child genome in as this genome's newest
        // child and as the head of the tracker's global genome list.
        void NewMutantGenome(TDerived *pChild)
        {
            TDerived *pCurrentFirstChild = m_pFirstChild;
            m_pFirstChild = pChild;
            if (pCurrentFirstChild != nullptr)
            {
                pCurrentFirstChild->m_pPrevSibling = pChild;
                pChild->m_pNextSibling = pCurrentFirstChild;
            }
            pChild->m_pNext = m_pTracker->m_pListFirst;
            if (m_pTracker->m_pListFirst != nullptr)
            {
                m_pTracker->m_pListFirst->m_pPrev = pChild;
            }
            m_pTracker->m_pListFirst = pChild;
        }

        // Unlinks this genome from both the sibling tree and the tracker's
        // global list, recursively pruning a parent that becomes a dead,
        // childless branch.
        void RemoveGenomeFromTree()
        {
            m_pTracker->m_nTreeGenomes--;
            TDerived *pLeftSib = m_pNextSibling;
            TDerived *pRightSib = m_pPrevSibling;
            TDerived *pParent = m_pParent;
            if ((pParent != nullptr) && (Self() == pParent->m_pFirstChild))
            {
                pParent->m_pFirstChild = pLeftSib;
                if ((pLeftSib == nullptr) && (pParent->m_pop == 0))
                {
                    if (pParent->m_alive)
                    {
                        pParent->m_alive = false;
                        m_pTracker->m_nLivingGenomes--;
                    }
                    pParent->RemoveGenomeFromTree();
                }
            }
            if (pRightSib != nullptr)
            {
                pRightSib->m_pNextSibling = pLeftSib;
            }
            if (pLeftSib != nullptr)
            {
                pLeftSib->m_pPrevSibling = pRightSib;
            }
            if (m_pPrev != nullptr)
            {
                m_pPrev->m_pNext = m_pNext;
            }
            if (m_pNext != nullptr)
            {
                m_pNext->m_pPrev = m_pPrev;
            }
            if (m_pTracker->m_pListFirst == Self())
            {
                m_pTracker->m_pListFirst = m_pNext;
            }
            if (m_pTracker->m_pListLast == Self())
            {
                m_pTracker->m_pListLast = m_pPrev;
            }
            m_pNext = nullptr;
            m_pPrev = nullptr;
        }

        // Marks this genome dead and, with leaf removal on, prunes it if it
        // is a leaf.
        void KillGenome()
        {
            m_alive = false;
            m_pTracker->m_nLivingGenomes--;
            if (m_pTracker->m_removeEmptyLeaves && (m_pFirstChild == nullptr))
            {
                RemoveGenomeFromTree();
            }
        }

        int64_t m_pop;
        int32_t m_id;
        TDerived *m_pParent;
        TDerived *m_pFirstChild;
        TDerived *m_pNextSibling;
        TDerived *m_pPrevSibling;
        TDerived *m_pNext;
        TDerived *m_pPrev;
        PhylogenyTracker<TDerived> *m_pTracker;
        bool m_alive;
    };

    template <typename TDerived>
    class PhylogenyTracker
    {
    public:
        // removeEmptyLeaves: whether dead, childless genomes are pruned
        // from the tree as they die (bounds tree size for long runs) or
        // kept (needed to record a full phylogeny over time).
        explicit PhylogenyTracker(bool removeEmptyLeaves)
            : m_genomes{},
              m_nGenomesEver(0),
              m_nLivingGenomes(0),
              m_nTreeGenomes(0),
              m_totalPop(0),
              m_removeEmptyLeaves(removeEmptyLeaves),
              m_pProgenitor(nullptr),
              m_pListFirst(nullptr),
              m_pListLast(nullptr),
              m_popRecords{},
              m_recordLabels{}
        {
        }

        ~PhylogenyTracker()
        {
        }

        PhylogenyTracker(const PhylogenyTracker &other) = delete;
        PhylogenyTracker &operator=(const PhylogenyTracker &other) = delete;

        // Creates and takes ownership of a new genome constructed as
        // TDerived(pParent, args...). Pass pParent == nullptr for the
        // progenitor, otherwise the genome it mutated from. The genome
        // starts alive with population 0; IncPop() adds its first
        // individual.
        template <typename... Args>
        TDerived &NewGenome(TDerived *pParent,
                            Args     &&...args)
        {
            if ((pParent == nullptr) && (m_pProgenitor != nullptr))
            {
                throw std::logic_error("PhylogenyTracker already has a progenitor; pass the parent genome");
            }
            m_genomes.emplace_back(pParent, std::forward<Args>(args)...);
            TDerived &genome = m_genomes.back();
            RegisterNewGenome(&genome, pParent);
            return genome;
        }

        TDerived *GetProgenitor() const
        {
            return m_pProgenitor;
        }

        int32_t GetNumGenomes() const
        {
            return m_nGenomesEver;
        }

        int32_t GetNumLivingGenomes() const
        {
            return m_nLivingGenomes;
        }

        int32_t GetNumTreeGenomes() const
        {
            return m_nTreeGenomes;
        }

        int64_t GetTotalPop() const
        {
            return m_totalPop;
        }

        bool RemovesEmptyLeaves() const
        {
            return m_removeEmptyLeaves;
        }

        // Runs fn(TDerived&) on every genome still in the tree, in creation
        // (id) order. Genomes pruned by leaf removal are skipped.
        template <typename GenomeFn>
        void ForEachGenome(GenomeFn fn) const
        {
            TDerived *pGenome = m_pListLast;
            while (pGenome != nullptr)
            {
                fn(*pGenome);
                pGenome = pGenome->m_pPrev;
            }
        }

        // Records a snapshot of every genome's current population at a named
        // timepoint, for later clone-size output (see OutputClonesToCSV).
        // Requires leaf removal to be off (a full phylogeny must be kept to
        // record it over time); the label must not contain a comma or be a
        // reserved column name.
        void RecordClones(const std::string &timepointLabel)
        {
            if (m_removeEmptyLeaves)
            {
                throw std::logic_error("PhylogenyTracker::RecordClones can't record populations with leaf removal on");
            }
            if (CountCommas(timepointLabel) > 0)
            {
                throw std::invalid_argument("PhylogenyTracker::RecordClones timepoint label can't contain a comma");
            }
            if ((timepointLabel == "CloneID") || (timepointLabel == "ParentID"))
            {
                throw std::invalid_argument("PhylogenyTracker::RecordClones timepoint label can't be CloneID or ParentID");
            }
            std::vector<int64_t> pops(static_cast<size_t>(m_nGenomesEver), 0);
            ForEachGenome([&](TDerived &genome)
            {
                pops[static_cast<size_t>(genome.GetId())] = genome.GetPop();
            });
            m_popRecords.push_back(std::move(pops));
            m_recordLabels.push_back(timepointLabel);
        }

        // Writes the recorded clone populations to a CSV suitable for a
        // Muller/clone-size plot. Columns: the caller's attribute headers,
        // then CloneID, ParentID, then one column per recorded timepoint.
        // attrsFn(const TDerived&) returns that clone's attributes as a
        // comma-separated string with exactly attrHeaders.size() fields.
        // Clones whose peak (subtree-summed) population never exceeds
        // excludePopCutoff are omitted and folded into their parent.
        template <typename AttrsFn>
        void OutputClonesToCSV(const std::string              &path,
                               const std::vector<std::string> &attrHeaders,
                               AttrsFn                         attrsFn,
                               int64_t                         excludePopCutoff)
        {
            if (excludePopCutoff < 0)
            {
                throw std::invalid_argument("PhylogenyTracker::OutputClonesToCSV pop cutoff must be >= 0");
            }
            for (const std::string &header : attrHeaders)
            {
                if (CountCommas(header) > 0)
                {
                    throw std::invalid_argument("PhylogenyTracker::OutputClonesToCSV attribute header can't contain a comma");
                }
                if ((header == "CloneID") || (header == "ParentID"))
                {
                    throw std::invalid_argument("PhylogenyTracker::OutputClonesToCSV attribute header can't be CloneID or ParentID");
                }
            }

            std::vector<int32_t> parentIDs = GenParentIDs();
            std::vector<std::vector<int64_t>> pops = SumPops(parentIDs);
            std::vector<int64_t> maxPops = ThreshPops(pops, parentIDs, excludePopCutoff);

            std::ofstream out(path);
            if (!out.is_open())
            {
                Logger::Instance().Log(LogSeverity::Error, "PhylogenyTracker::OutputClonesToCSV failed to open file for writing: " + path);
                throw std::runtime_error("failed to open file for writing: " + path);
            }

            out << JoinStrings(attrHeaders) << ",CloneID,ParentID," << JoinStrings(m_recordLabels) << "\n";

            int32_t expectedCommas = static_cast<int32_t>(attrHeaders.size()) - 1;
            ForEachGenome([&](TDerived &clone)
            {
                int32_t id = clone.GetId();
                if (maxPops[static_cast<size_t>(id)] <= excludePopCutoff)
                {
                    return;
                }
                std::string cloneAttrs = attrsFn(clone);
                if (CountCommas(cloneAttrs) != expectedCommas)
                {
                    throw std::invalid_argument("PhylogenyTracker::OutputClonesToCSV clone attributes must have one field per attribute header");
                }
                out << cloneAttrs << "," << id << "," << parentIDs[static_cast<size_t>(id)] << ",";
                for (size_t record = 0; record < pops.size(); record++)
                {
                    int64_t pop = 0;
                    if (static_cast<int32_t>(pops[record].size()) > id)
                    {
                        pop = pops[record][static_cast<size_t>(id)];
                    }
                    out << pop;
                    if (record < (pops.size() - 1))
                    {
                        out << ",";
                    }
                }
                out << "\n";
            });
        }

    private:
        friend class Genome<TDerived>;

        static int32_t CountCommas(const std::string &text)
        {
            return static_cast<int32_t>(std::count(text.begin(), text.end(), ','));
        }

        static std::string JoinStrings(const std::vector<std::string> &parts)
        {
            std::string joined;
            for (size_t i = 0; i < parts.size(); i++)
            {
                joined += parts[i];
                if (i < (parts.size() - 1))
                {
                    joined += ",";
                }
            }
            return joined;
        }

        // parentIDs[id] = the id of that genome's parent (0 for the
        // progenitor, which has no parent).
        std::vector<int32_t> GenParentIDs() const
        {
            std::vector<int32_t> parentIDs(static_cast<size_t>(m_nGenomesEver), 0);
            ForEachGenome([&](TDerived &genome)
            {
                if (genome.GetId() != 0)
                {
                    parentIDs[static_cast<size_t>(genome.GetId())] = genome.GetParent()->GetId();
                }
            });
            return parentIDs;
        }

        // Copies the pop records and accumulates each clone's population up
        // into its parent, so each entry becomes an inclusive subtree total
        // (children, with higher ids, are summed before their parents).
        std::vector<std::vector<int64_t>> SumPops(const std::vector<int32_t> &parentIDs) const
        {
            std::vector<std::vector<int64_t>> result = m_popRecords;
            for (std::vector<int64_t> &pops : result)
            {
                for (int32_t i = static_cast<int32_t>(pops.size()) - 1; i > 0; i--)
                {
                    pops[static_cast<size_t>(parentIDs[static_cast<size_t>(i)])] += pops[static_cast<size_t>(i)];
                }
            }
            return result;
        }

        // Zeroes clones whose peak inclusive population never exceeds
        // cutoff, then breaks the kept inclusive totals back down
        // (subtracting each clone from its parent) so surviving siblings'
        // bands stack. Returns the per-clone peak populations.
        std::vector<int64_t> ThreshPops(std::vector<std::vector<int64_t>> &popsArr,
                                        const std::vector<int32_t>        &parentIDs,
                                        int64_t                            cutoff) const
        {
            std::vector<int64_t> maxPops(static_cast<size_t>(m_nGenomesEver), 0);
            for (const std::vector<int64_t> &pops : popsArr)
            {
                for (size_t i = 0; i < pops.size(); i++)
                {
                    maxPops[i] = std::max(maxPops[i], pops[i]);
                }
            }
            for (size_t i = 0; i < maxPops.size(); i++)
            {
                if (maxPops[i] <= cutoff)
                {
                    for (std::vector<int64_t> &pops : popsArr)
                    {
                        if (pops.size() > i)
                        {
                            pops[i] = 0;
                        }
                    }
                }
            }
            for (std::vector<int64_t> &pops : popsArr)
            {
                for (size_t j = 1; j < pops.size(); j++)
                {
                    pops[static_cast<size_t>(parentIDs[j])] -= pops[j];
                }
            }
            return maxPops;
        }

        void RegisterNewGenome(TDerived *pChild,
                               TDerived *pParent)
        {
            pChild->m_pTracker = this;
            pChild->m_alive = true;
            pChild->m_id = m_nGenomesEver;
            m_nGenomesEver++;
            m_nLivingGenomes++;
            m_nTreeGenomes++;
            pChild->m_pop = 0;
            if (pParent != nullptr)
            {
                pChild->m_pParent = pParent;
                pParent->NewMutantGenome(pChild);
            }
            else
            {
                m_pProgenitor = pChild;
                m_pListFirst = pChild;
                m_pListLast = pChild;
            }
        }

        std::deque<TDerived> m_genomes;
        int32_t m_nGenomesEver;
        int32_t m_nLivingGenomes;
        int32_t m_nTreeGenomes;
        int64_t m_totalPop;
        bool m_removeEmptyLeaves;
        TDerived *m_pProgenitor;
        TDerived *m_pListFirst;
        TDerived *m_pListLast;
        std::vector<std::vector<int64_t>> m_popRecords;
        std::vector<std::string> m_recordLabels;
    };
}
