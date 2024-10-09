import numpy as np
from arsenal.maths import logsumexp
from collections import defaultdict
from tokenization.util import flatten


class Item:
    __slots__ = ('ps', 'ys', 'offset', 'parent')
    def __init__(self, ps, ys, offset, parent):
        self.ps = ps
        self.ys = ys
        self.offset = offset
        self.parent = parent
    def __repr__(self):
        return f'Item(ps={self.ps!r}, ys={self.ys!r})'
    def __eq__(self, other):
        return self.ps == other.ps and self.ys == other.ys
    def __hash__(self):
        return hash(self.ys)


class Beam(list):

    def __init__(self, *args):
        super().__init__(*args)
        self.parent = None

    @property
    def total(self):
        return self.logsumexp()

    def logsumexp(self):
        return logsumexp([x.ps for x in self])

    @property
    def fsa(self):
        from fsa import FSA
        tmp = FSA()
        for item in self:
            tmp += FSA.from_string(flatten(item.ys))
        return tmp.min()

    def sort(self):
        return Beam(sorted(self, key = lambda item: -item.ps))

    def top(self, K):
        return Beam(self.sort()[:K])

    def argmax(self):
        return max(self, key = lambda item: -item.ps)

    def __repr__(self):
        return '%s (%r/%s) {\n%s\n}' % (
            self.__class__.__name__,
            getattr(self, 'key', None),
            logsumexp([x.ps for x in self]) if len(self) > 0 else -np.inf,
            '\n'.join(f'  {x},' for x in self)
        )

    def depth(self):
        return max([len(item.ys) for item in self], default=0)

    def groupby(self, key):
        groups = defaultdict(Beam)
        for item in self:
            groups[key(item)].append(item)
        return groups

    def backup(self, qs):
        C = set()
        N = len(qs)
        for item in self:
            #if qs == ''.join(item.ys):  # no need to backup a perfect cover without any slop
            if N == item.offset:
                C.add(item)
            else:
                C.add(item.parent)
        return Beam(C)
