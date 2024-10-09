import html
import pandas as pd
import numpy as np
import pylab as pl
from collections import namedtuple
from time import time
from functools import cached_property
from IPython.display import display

from arsenal import Integerizer, colors, iterview
from arsenal.maths import sample_dict

from tokenization.lm import load_model_by_name
from tokenization.util import strict_prefix, covers, flatten, prefixes, Chart, display_table
from tokenization.basics import Beam, Item


MemoKey = namedtuple('MemoKey', 'signature, tainted')


class beam_search:
    def __init__(self, llm, qs, K, init=None, progress=False):
        self.llm = llm

        self.K = K
        self.qs = qs
        if init is None:
            beam = Beam([Item(ps=0, ys=(), offset=0, parent=None)])
            beam.parent = None
            beam.key = MemoKey(signature='', tainted=False)
        else:
            beam = init
        beams = [beam]
        t = 0
        while True:
            t += 1
            b4 = time()
            key, next_beam = self.next_beam(beam, qs)
            if progress:
                print(f'step: {t}, took {time() - b4:.3f} sec')
            next_beam.parent = beam
            beam = next_beam
            beam.key = key     # TODO: this is a little hacky
            if not beam:
                break
            beams.append(beam)
        complete = Beam()
        for beam in beams:
            # TODO: rename "tainted"
            if not beam.key.tainted: continue
            for item in beam:
                if covers(qs, item.ys):
                    complete.append(item)
        self._beams = beams
        self._init = init
        self.final = complete.top(self.K)

    def next_beam(self, beam, qs):
        key, candidates = self.extend(beam, qs)
        return key, candidates.top(self.K)

    @cached_property
    def beams(self):
        earlier = []
        curr = self._init
        while curr is not None:
            earlier.append(curr)
            curr = curr.parent
        earlier.reverse()
        #print(colors.light.red % '>>>>>', earlier)
        beams = self._beams
        if earlier:
            beams = earlier + beams[1:]
        return beams

    def extend(self, beam, qs):
        """
        Extend the `beam` so that it either covers or makes progress on covering the
        string `qs`.
        """

        L = 0

        candidates = Beam()
        tainted = False

        nqs = len(qs)

        # TODO: batch the p_next evaluation of the entire beam
        for item in beam:
            offset = item.offset

            if offset >= nqs:
                continue

            logp = self.llm.logp_next(item.ys)

            qqs = qs[offset:]
            nqqs = nqs - offset

            I = 0

            # TODO: speedup ideas
            #   1) trie-structured `p`
            #   2) sorted p - sorting will allow more efficient pruning on the prefix
            #   3) pandas has some vectorized string search methods
            # TODO: skip if the item can't be in the top K anyways
            # TODO: avoid creating `Item` all together, instead create a mask vector

            for y, logpy in logp.items():

                # Determined the first point of difference
                ny = len(y)
                ub = min(nqqs, ny)

                i = 0
                while i < ub and y[i] == qqs[i]:
                    i += 1

                if i > I:
                    I = i

                C = (i == nqqs)
                P = (i == ny)

                if C:
                    tainted = True

                if C or P:
                    new_item = Item(
                        ps = item.ps + logpy,
                        offset = offset + ny,
                        ys = (item.ys, y),
                        parent = item,
                    )
                    candidates.append(new_item)

            L = max(L, I + offset)

        key = MemoKey(signature=qs[:L+1], tainted=tainted)

        return key, candidates

    def __repr__(self):
        return '%s.%s' % (self.__class__.__name__, self.final)

    def tree(self):
        root = Node()
        for beam in self.beams:
            for item in beam:
                node = root.find(flatten(item.ys))
                node.data = item
        return root

    def graphviz(self, fmt_node=lambda x: f'{x.data.ps:.1f}', **kwargs):
        return self.tree().graphviz(fmt_node=fmt_node, **kwargs)

    def compare(self, B, stop_early=False):
        if len(self.beams) != len(B.beams):
            print(colors.light.red % 'DIFFERENT LENGTHS')
        for t, (a, b) in enumerate(zip(self.beams, B.beams)):
            print()
            print(colors.mark(a == b), colors.light.blue % 'STEP', t)
            if a == b:
                print(a)
            else:
                print('DIFFERENCES', f'(lengths: {(len(a), len(b))})')
                display_table([[a, b]])
                if stop_early:
                    return [a,b]
        return [None, None]

    def find_common_prefix(self, B, verbose=False, noend=False):
        assert self.beams[0] == B.beams[0]
        last = None
        for t, (a, b) in enumerate(zip(self.beams, B.beams)):
            if a == b:
                if noend and a.key[1]:
                    continue
                if verbose:
                    print()
                    print(colors.mark(a == b), colors.light.blue % 'STEP', t)
                    print(a)
                last = a
            else:
                break
        return last


#class beam_search_vectorized(beam_search):
#    def __init__(self, llm, qs, K, init=None, progress=False):
#        self.vocabulary = np.array(llm._decode)
#
#        self.vocabulary_sort = np.sort(self.vocabulary)
#        self.vocabulary_argsort = np.argsort(self.vocabulary)
#
#        self.lens = np.array([len(x) for x in self.vocabulary])
#        self.df = pd.DataFrame({'surface_form': np.array(llm._decode)})
#        super().__init__(llm, qs, K, init=init, progress=progress)
#
#    def next_beam(self, beam, qs):
#        key, candidates = self.extend_vectorized(beam, qs)
#
#        flat_arr = candidates.ravel()
#
#        # Get the coordinates of the top K values
#        indices = np.argsort(flat_arr)[-self.K:]
#        top_coords = np.array(np.unravel_index(indices, candidates.shape)).T
#        top_values = flat_arr[indices]
#
#        next_beam = Beam()
#
#        for (item_id, i), value in zip(top_coords, top_values):
#            if value == -np.inf: continue
#            item = beam[item_id]
#            y = self.vocabulary[i]
#            next_beam.append(Item(
#                ps = value,
#                offset = item.offset + len(y),
#                ys = (item.ys, y),
#                parent = item,
#            ))
#
#        return key, next_beam
#
#    def extend_vectorized(self, beam, qs):
#        """
#        Extend the `beam` so that it either covers or makes progress on covering the
#        string `qs`.
#        """
#
#        L = 0
#
#        tainted = False
#
#        nqs = len(qs)
#
#        candidates = np.full((len(beam), len(self.vocabulary)), -np.inf)
#
#        # TODO: batch the logp_next evaluation of the entire beam
#        for item_id, item in enumerate(beam):
#            offset = item.offset
#
#            if offset >= nqs:
#                continue
#
#            logp = self.llm.logp_next(item.ys)
#
#            qqs = qs[offset:]
#            nqqs = nqs - offset
#
#            prefix = qqs
#            arr = self.vocabulary_sort
#            # Find the start index using searchsorted (binary search)
#            start_idx = arr.searchsorted(prefix)
#
#            # Find the end of the prefix range by incrementing the last character of the prefix
#            # Create a hypothetical string that is just greater than all strings starting with 'prefix'
#            # e.g., for 'pre', the next string would be 'pre{'
#            end_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
#            end_idx = arr.searchsorted(end_prefix)
#
#            mask_cover = np.full(len(self.vocabulary), False)
#            mask_cover[self.vocabulary_argsort[start_idx:end_idx]] = True
#
#            #want_mask_cover = self.df.surface_form.str.startswith(qqs)
#            #assert np.all(want_mask_cover == mask_cover), embed()
#
#            mask_prefix = self.df.surface_form.apply(qqs.startswith)
#            #Ys = self.vocabulary[mask_cover | mask_prefix]
#
#            tainted = tainted or mask_cover.any()
#
#            active = mask_cover | mask_prefix
#
#            candidates[item_id, :] = item.ps + logp.values()
#            candidates[item_id, ~active] = -np.inf
#
#            #ns = self.lens + item.offset
#            I = min(nqqs, self.lens[active].max())
#            L = max(L, I + offset)
#
#        key = MemoKey(signature=qs[:L+1], tainted=tainted)
#
#        return key, candidates


#def find_prefix_indices(arr, prefix):
#    # Find the start index using searchsorted (binary search)
#    start_idx = np.searchsorted(arr, prefix)
#
#    # Find the end of the prefix range by incrementing the last character of the prefix
#    # Create a hypothetical string that is just greater than all strings starting with 'prefix'
#    # e.g., for 'pre', the next string would be 'pre{'
#    end_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1)
#    end_idx = np.searchsorted(arr, end_prefix)
#
#    return np.arange(start_idx, end_idx)


#beam_search = beam_search_vectorized


#def new(llm, context, K, verbose=False):
#    """
#    Approximate p(next_character | context) where `context` is a string and `K`
#    is the beam search parameter.
#    """
#    init = backup_beam(context, beam_search(llm, context, K).final)
#    if verbose:
#        print(colors.yellow % 'initial beam:', init)
#    return next_char_distribution(llm, context, init)


# TODO: compute probabilities in log space
class Estimator:

    def __init__(self, llm, K):
        self.llm = llm
        self.K = K
        self.tmbs = MemoizedBeamSearch(llm, K=K)
        self.cache = {}

    def p_next(self, context):
        result = self.cache.get(context)
        if result is not None:
            return result
        beam = self.tmbs(context).final
        init = beam.backup(context)
        result = self._p_next_backup(context, init)
        self.cache[context] = result
        return result

    def p_next_string(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        P = 1
        for i in range(len(extension)):
            p = self.p_next(context + extension[:i])
            P *= p[extension[i]]
        return P

    def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.p_next(context)
            if verbose:
                print(context, p.top(5))
            x = p.argmax()
            context += x
        return context

    def sample(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.p_next(context)
            if verbose:
                print(context, p.top(5))
            x = sample_dict(p)
            context += x
        return context

    def _p_next_direct(self, context, beam):
        r"""
        Similar to `next_char_distribution` except we do not back up the `beam` by
        one token.

        Warning: This is an inaccurate estimator when the beam size is not really large.
        """
        Q = Chart(0)
        N = len(context)
        for item in beam:
            # TODO: we can avoid this concatenation by figuring out where in the
            # last token we are. This is a simple function of the offsets and N.
            xs = ''.join(flatten(item.ys))
            assert covers(context, item.ys), item.ys
            if len(xs) <= N: continue  # TODO: this is kind of ugly; we should extend it
            # TODO: work in log space (i.e., use logsumexp and save
            # exponentiation until the end).
            Q[xs[N]] += np.exp(item.ps)
        return Q.normalize()

    def _p_next_backup(self, context, beam):
        r"""
        Approximate `p(next_character | context)` where `context` is a string and
        beam is collection of token strings that are one token away from covering
        `context next_character` for all `next_character`s.

        Estimate the conditional distribution of single-character extension of the
        string `context` ($\boldsymbol{\sigma}'$ in the equation below):
        $$
        \widehat{p}(\sigma \mid \texttt{context})
        = \frac{\widehat{w}(\texttt{context} \, \sigma) }
               {\sum_{\sigma'} \widehat{w}(\texttt{context} \, \sigma')}
        $$

        Note that this method does not assume `beam` is exhaustive covering; when
        `beam` is not exhaustive the inferred distribution will only be
        approximation.
        """
        #assert isinstance(context, str) and isinstance(beam, Beam)
        Q = Chart(0)
        N = len(context)
        for item in beam:
            logp = self.llm.logp_next(item.ys)
            offset = item.offset
            n = N - offset
            #print()
            #print('context=', repr(context))
            #print(item)
            #assert context[:offset] == ''.join(flatten(item.ys))
            qqs = context[offset:]
            for y, logpy in logp.items():
                if strict_prefix(qqs, y):
                    # TODO: work in log space (i.e., use logsumexp and save
                    # exponentiation until the end).
                    Q[y[n]] += np.exp(item.ps + logpy)
        return Q.normalize()


def canonical_beam(llm, context):
    raise NotImplementedError()

    # TODO: need to use cons approach
#    canonical = llm.encode_prompt(context)
#
#    item = Item(ps=0, ys=(), offset=0, parent=None)
#    beam = Beam([item])
#    beam.parent = None
#
#    for t in range(len(canonical)):
#        y = canonical[t]
#        next_item = Item(
#            ps=item.ps + llm.logp_next(canonical[:t])[y],
#            ys=(item.ys, y),
#            offset = item.offset + len(y),
#            parent = item,
#        )
#        next_beam = Beam([next_item])
#        next_beam.parent = beam
#        beam = next_beam
#        item = next_item
#
#    return beam


class MemoizedBeamSearch:

    def __init__(self, llm, K):
        self.llm = llm
        self.root = Node()
        self.K = K
        # initialize the cache with the bottom-most memo
        out = beam_search(self.llm, '', K=self.K)
        # Note: we only add the *new* memos
        self.add_memos(out._beams)

    def fresh(self, qs):
        "Run beam_search without memoization."
        return beam_search(self.llm, qs, K=self.K)

    def find_memo(self, qs):
        for node in reversed(list(self.root.find_all(qs))):
            for beam in node.data:
                assert not beam.key.tainted    # Note: 'tainted' memos have already been filtered out
#                assert not (beam.parent is not None and beam.parent.key.signature == beam.key.signature)
                if beam.parent is not None and beam.parent.key.signature == beam.key.signature:
                    continue
                return beam

    def __call__(self, qs, update=True):
        init = self.find_memo(qs)
        out = beam_search(self.llm, qs, K=self.K, init=init)
        if update:
            # Note: we only add the *new* memos
            for beam in out._beams:
                self.update(beam)
        return out

    def graphviz(self, fmt_node=lambda x: str(len(x.data)) if x.data else ' ', **kwargs):
        return self.root.graphviz(fmt_node=fmt_node, **kwargs)

    def update(self, beam):
        # what is the largest portion of the string that has been covered?
        self.add_memos([beam])

    def add_memos(self, memos):
        for beam in memos:
            node = self.root.find(beam.key[0])
            if beam.key.tainted: continue             # XXX: drop these memo as we cannot use them
#            if beam.parent is not None and beam.parent.key.signature == beam.key.signature:
#                continue
            if beam not in node.data:
                node.data.append(beam)


class Node:

    def __init__(self):
        self.data = []
        self.children = {}

    def __getitem__(self, key):
        return self.children[key]

    def find(self, word):
        curr = self
        for letter in word:
            if letter not in curr.children:
                curr.children[letter] = Node()
            curr = curr.children[letter]
        return curr

    def find_all(self, word):
        curr = self
        yield curr
        for letter in word:
            if letter not in curr.children:
                break
            curr = curr.children[letter]
            yield curr

    def __repr__(self):
        return f'Node({self.data!r})'

    def graphviz(
        self,
        fmt_edge=lambda x, a, y: f'{html.escape(str(a).replace(" ", "␣"))}',
        fmt_node=lambda x: f'{html.escape(str(x))}',
    ):
        "Create a graphviz instance for this subtree"
        from graphviz import Digraph
        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='10',
                height='.05',
                width='.05',
                margin='0.055,0.042',
            ),
            edge_attr=dict(arrowsize='0.3', fontname='Monospace', fontsize='9'),
        )
        f = Integerizer()
        xs = set()
        q = [self]
        while q:
            x = q.pop()
            xs.add(x)
            if x.children is None:
                continue
            for a, y in x.children.items():
                g.edge(str(f(x)), str(f(y)), label=f'{fmt_edge(x,a,y)}')
                q.append(y)
        for x in xs:
            g.node(str(f(x)), label=str(fmt_node(x)), shape='box')
        return g


#_______________________________________________________________________________
# Tests


def test_covers():
    assert covers('The', ((), 'The',))
    assert covers('The', ((), 'There',))
    assert not covers('The', (((), 'The'), ' following'))
    assert covers('There are no', ((((), 'There'), ' are'), ' now'))


def test_memoized_beam_search():

    llm = load_model_by_name('gpt2')

    mbs = MemoizedBeamSearch(llm, K=8)

    qs = 'There are '
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)

    qs = 'There are many'
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)

    qs = 'Therefore'
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)

    # running the same query string another time leads to trickier cases
    qs = 'Therefore'
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)

    qs = 'Therefore, I disagree'
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)

    qs = 'There are no clear answers'
    print(colors.cyan % repr(qs))
    warm = mbs(qs)
    fresh = beam_search(llm, qs, K=mbs.K)
    _test_memoized_beam_search(warm, fresh)


def _test_memoized_beam_search(warm, fresh):

    have = warm.final
    want = fresh.final

    if have == want:
        print(colors.ok)

        #print(colors.light.blue % 'have:', have)
        #print(colors.light.blue % 'want:', want)

    else:

        print('fresh beams:')
        display(fresh.beams)

        print('warm beams:')
        display(warm.beams)

    assert have == want


def test_profile():
    from arsenal.profiling import profiler
    llm = load_model_by_name('gpt2')

    stimulus = 'Therefore, I am not impressed with the speed up.'

    tmbs = MemoizedBeamSearch(llm, 5)

    with profiler():
        for i in iterview(range(0, len(stimulus))):
            qs = stimulus[:i+1]
            tmbs(qs)


def test_compare_speeds():
    from arsenal import timers
    from arsenal.viz import update_ax

    stimulus = 'Therefore, I am not impressed with the speed up.'

    # Note: we create two copies of the llm so that the running time isn't skewed by caching
    llm1 = load_model_by_name('gpt2')
    llm2 = load_model_by_name('gpt2')

    tmbs1 = MemoizedBeamSearch(llm1, 5)
    tmbs2 = MemoizedBeamSearch(llm2, 5)

    pl.ion()  # Enable interactive mode
    fig = pl.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    T = timers()
    for qs in iterview(list(prefixes(stimulus))):
        if len(qs) == 0: continue
        with T['memo'](N=len(qs)):
            have = tmbs1(qs)
        with T['fresh'](N=len(qs)):
            want = tmbs2.fresh(qs)
        with update_ax(ax):
            T.plot_feature('N', ax=ax, show_curve=(len(qs) > 2))
            #ax.loglog()
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))  # move legend outside
            ax.set_title(repr(qs))
            #ax.set_xticks(range(len(qs)+1))
            #ax.set_xticklabels([''] + list(qs.replace(' ', '␣')));
        assert have.final == want.final

    pl.ioff()    # Disable interactive mode after the loop
    pl.show()


def test_run():
    stimulus = 'Therefore, I am not impressed with the speed up.'
    llm = load_model_by_name('gpt2')
    tmbs = MemoizedBeamSearch(llm, 5)
    for qs in iterview(list(prefixes(stimulus))):
        tmbs(qs)


def test_generate():
    llm = load_model_by_name('gpt2-large')
    M = Estimator(llm, K=5)
    context = 'An apple a day keeps the '
    have = M.greedy(context, steps=200, verbose=True)
    print(have)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
