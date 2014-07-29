#ifndef DISJOINTSET_H_INCLUDED
#define DISJOINTSET_H_INCLUDED

#include <vector>

class CDisjointSet
{
public:
        CDisjointSet();

        CDisjointSet(int count);

        CDisjointSet(CDisjointSet& s);

        ~CDisjointSet();

        void makeset(int count);

        int find(int element);

        void merge(int elementA, int elementB);

        void subSet(std::vector<std::vector<int> >& sets);
private:
        struct node
        {
                int rank;
                int parent;

                node()
                {
                        rank = 0;
                        parent = -1;
                }
        };
        std::vector<node> m_nodes;
};


#endif // DISJOINTSET_H_INCLUDED
