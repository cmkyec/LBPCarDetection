#include "disjointSet.h"
#include <cassert>

CDisjointSet::CDisjointSet()
{
        makeset(0);
}

CDisjointSet::CDisjointSet(int count)
{
        makeset(count);
}

CDisjointSet::CDisjointSet(CDisjointSet& s)
{
        m_nodes = s.m_nodes;
}

CDisjointSet::~CDisjointSet()
{
        std::vector<node>().swap(m_nodes);
}

void CDisjointSet::makeset(int count)
{
        m_nodes.resize(count);
        for (std::size_t i = 0; i < m_nodes.size(); ++i) {
                m_nodes[i] = node();
        }
}

int CDisjointSet::find(int element)
{
        assert(element < m_nodes.size());

        int root = (int)element;
        while (m_nodes[root].parent != -1) {
                root = m_nodes[root].parent;
        }
        // path compression
        while (m_nodes[element].parent >= 0 && m_nodes[element].parent != root) {
                int curParent = m_nodes[element].parent;
                m_nodes[element].parent = root;
                element = curParent;
        }

        return root;
}

void CDisjointSet::merge(int elementA, int elementB)
{
        assert(elementA < m_nodes.size());
        assert(elementB < m_nodes.size());

        if (elementA == elementB) return;

        int rootA = find(elementA);
        int rootB = find(elementB);
        if (rootA == rootB) return;

        int rankA = m_nodes[rootA].rank;
        int rankB = m_nodes[rootB].rank;
        if (rankA > rankB) {
                m_nodes[rootB].parent = rootA;
        } else {
                m_nodes[rootA].parent = rootB;
                m_nodes[rootB].rank += (rankA == rankB);
        }
}

void CDisjointSet::subSet(std::vector<std::vector<int> >& sets)
{
        std::vector<node> nodes = m_nodes;
        std::vector<int> label(nodes.size());
        int nlabel = 0;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
                int root = find((int)i);
                if (nodes[root].rank >= 0) nodes[root].rank = ~nlabel++;
                label[i] = ~nodes[root].rank;
        }

        sets.resize(nlabel);
        for (std::size_t i = 0; i < label.size(); ++i) {
                sets[label[i]].push_back((int)i);
        }
}




