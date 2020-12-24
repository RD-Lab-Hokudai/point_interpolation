#pragma once
using namespace std;

struct UnionFind
{
    int d[1000 * 1000];
    UnionFind(int n = 0)
    {
        for (int i = 0; i < n; i++)
            d[i] = -1;
    }
    int root(int x)
    {
        if (d[x] < 0)
            return x;
        return d[x] = root(d[x]);
    }
    bool unite(int x, int y)
    {
        x = root(x);
        y = root(y);
        if (x == y)
            return false;
        if (d[x] > d[y])
            swap(x, y);
        d[x] += d[y];
        d[y] = x;
        return true;
    }
    bool same(int x, int y) { return root(x) == root(y); }
    int size(int x) { return -d[root(x)]; }
};