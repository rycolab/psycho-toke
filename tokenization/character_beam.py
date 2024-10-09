import numpy as np
import pylab as pl
import seaborn as sns
from time import time
from arsenal import iterview, timers, timeit, colors
from arsenal.maths import sample_dict, logsumexp
from arsenal.iterextras import batch
from collections import defaultdict
from numpy import logaddexp

from tokenization.basics import Item, Beam
from tokenization.lm import load_model_by_name
from tokenization.util import Chart, prefixes


def plot_surprisals(context, surprisals, batch_size=75):

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    assert len(context) == len(surprisals)
    N = len(surprisals)
    T = batch_size

    context = np.array(list(context.replace(' ', '␣')))
    surprisals = np.array(surprisals)
    for B in batch(batch_size, range(len(context))):

        fig = pl.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)

        sns.barplot(surprisals[B], ax=ax)

        #ax.set_title(repr(context))
        ax.set_xticks(range(len(context[B])))
        ax.set_xticklabels(list(context[B]))
        ax.set_ylabel('suprisal')

        sns.despine()


class character_beam1:

    def __init__(self, llm, K):
        self.llm = llm
        self.K = K
        self.cache = {
            '': Beam([Item(ps=0, ys=(), offset=0, parent=None)]),
        }

        self.ix = defaultdict(list)
        self.vocab = self.llm._decode
        for i, y in enumerate(self.vocab):
            self.ix[y[0]].append(i)
        self.lens = np.array([len(y) for y in self.vocab])

    def __call__(self, qs):

        result = self.cache.get(qs)
        if result is not None:
            return result

        N = len(qs)
        beam = self(qs[:-1])
        curr_char = qs[-1]

        #print(qs[:-1])

        #print()
        #print(qs)
        candidates = Beam()
        n_expanded = 0

        for item in beam:  # Note: already sorted
            if item.offset < N:
                if n_expanded < self.K:
                    n_expanded += 1
                    #print('#', end='')
                    logp = self.llm.logp_next(item.ys)
                    _ps = item.ps + logp._p
                    _offset = item.offset + self.lens
                    for i in self.ix[curr_char]:
                        candidates.append(Item(
                            ps = _ps[i],
                            offset = _offset[i],
                            ys = (item.ys, self.vocab[i]),
                            parent = item,
                        ))
            else:
                #print('.', end='')
                next_char = item.ys[1][N - item.parent.offset - 1]
                if next_char == curr_char:
                    candidates.append(item)

        #result = candidates.sort()

        # sort by total bucket weight
        buckets = candidates.groupby(key=lambda item: item.ys[0])
        top_K_buckets = sorted(buckets.values(), key=lambda bucket: -bucket.logsumexp())[:self.K]
        result = Beam()
        for bucket in top_K_buckets:
            result.extend(bucket)
        result = result.sort()

        assert len(result) <= self.K * len(self.vocab)

        self.cache[qs] = result
        return result


# Implementation note: The idea behind this implementation is to maintain a
# cover that has one character as a wildcard.  (That is essentially what we are
# doing in practice to get the other implementation of logp_next.)  This version
# extends an `item` when `item.offset == N`.  Note that more items will go on
# the beam as this case of extension doesn't filter by a character (i.e., it is
# unconstrained just like in the first step).  That said, those cases appear to
# just pop up in logp_next so it's probably equally fast.  The benefit of the
# approach in the implementation below is that the candidates will over all
# single-character extension of `qs`; Thus, it work directly with the logp_next
# logic (i.e., there is no need to extend the `item.offset == N` cases as we
# currently do in that method.
class character_beam2:

    def __init__(self, llm, K):
        self.llm = llm
        self.K = K
        self.vocab = self.llm._decode
        self.lens = np.array([len(y) for y in self.vocab])
        self.cache = {}

    def __call__(self, qs):

        result = self.cache.get(qs)
        if result is not None:
            return result

        N = len(qs)

        if N == 0:
            buckets = [Beam([Item(ps=0, ys=(), offset=0, parent=None)])]
        else:
            _buckets = defaultdict(Beam)
            curr_char = qs[-1]
            for item in self(qs[:-1]):
                last_token = item.ys[1]
                next_char = last_token[N - item.parent.offset - 1]
                if next_char == curr_char:
                    if item.offset > N:
                        _buckets[item.ys[0]].append(item)
                    else:
                        _buckets[item.ys].append(item)
            buckets = _buckets.values()

        # sort by total bucket weight
        top_K_buckets = sorted(buckets, key=lambda bucket: -bucket.logsumexp())[:self.K]
        result = Beam()
        for bucket in top_K_buckets:
            if len(bucket) == 1 and bucket[0].offset == N:
                item = bucket[0]
                logp = self.llm.logp_next(item.ys)
                ps = item.ps + logp._p
                offset = item.offset + self.lens
                ys = item.ys
                for i in range(len(self.vocab)):
                    _ps = ps[i]
                    _offset = offset[i]
                    _ys = (ys, self.vocab[i])
                    _item = Item(ps = _ps, offset = _offset, ys = _ys, parent = item)
                    result.append(_item)

            else:
                result.extend(bucket)

        #assert len(result) <= self.K * len(self.vocab)

        self.cache[qs] = result
        return result


class character_beam_estimator:

    def __init__(self, llm, K, backend=2):
        self.llm = llm
        if backend == 1:
            self.beam = character_beam1(llm, K)
        else:
            self.beam = character_beam2(llm, K)
        self.cache = {}

    def logp_next(self, context):
        result = self.cache.get(context)
        if result is not None:
            return result
        result = self._logp_next(context)
        self.cache[context] = result
        return result

    def _logp_next(self, context):
        candidates = self.beam(context)
        Q = Chart(-np.inf)
        N = len(context)
        for item in candidates:
            #assert item.offset >= N
            # Note: this case cannot be triggered in the character_beam2
            if item.offset == N:
                # extend to include at least one more character
                logp = self.llm.logp_next(item.ys)
                for y, logpy in logp.items():
                    next_char = y[0]
                    Q[next_char] = logaddexp(Q[next_char], item.ps + logpy)
            else:
                next_char = item.ys[1][N - item.parent.offset]
                Q[next_char] = logaddexp(Q[next_char], item.ps)
        Z = logsumexp(list(Q.values()))
        A = Chart(-np.inf)
        for next_char in Q:
            A[next_char] = Q[next_char] - Z
        return A

    def logp_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        P = 0
        for i in range(len(extension)):
            logp = self.logp_next(context + extension[:i])
            logP += p[extension[i]]
        return logP

    def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.logp_next(context)
            if verbose:
                print(repr(context), p.top(5))
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
                print(repr(context), p.top(5))
            x = sample_dict(p)
            context += x
        return context


def test_new_character_beam():
    llm = load_model_by_name('gpt2')
    K = 5

#    qs = 'Therefore, I am unimpressed with the speedup.'
    qs = 'Therefore, I am unimpressed.'

    T = timers()

    logp1 = {}
    logp2 = {}

    llm.clear_cache()
    M = character_beam_estimator(llm, K, backend=1)
    for context in iterview(list(prefixes(qs))):
        with T['v1'](N=len(context)):
            logp1[context] = M.logp_next(context)

    llm.clear_cache()
    M = character_beam_estimator(llm, K, backend=2)
    for context in iterview(list(prefixes(qs))):
        with T['v2'](N=len(context)):
            logp2[context] = M.logp_next(context)


    for context in prefixes(qs):

        err = sum(abs(np.exp(logp1[context][x]) - np.exp(logp2[context][x]))
                  for x in logp1[context].keys() | logp2[context].keys())

        tol = 0.001
        if err > tol:
            print(context)
            print(logp1[context].top(5))
            print(logp2[context].top(5))
            print(colors.mark(err <= tol), err)

        assert err <= tol

    from tokenization.prefix_cover import Estimator

    llm.clear_cache()
    M = Estimator(llm, K)
    for context in iterview(list(prefixes(qs))):
        with T['old'](N=len(context)):
            M.p_next(context)

    T.plot_feature('N')
    pl.show()


def test_generate():
    llm = load_model_by_name('gpt2-large')
    K = 5
    qs = 'An apple a day keeps the '
    M = character_beam_estimator(llm, K)
    with timeit('greedy generation'):
        output = M.greedy(qs, steps=12, verbose=True)
    print(repr(output))
    assert output == 'An apple a day keeps the doctor away.'


def test_profile():
    from arsenal.profiling import profiler

    llm = load_model_by_name('gpt2')
    K = 5

    qs = 'Therefore, I am unimpressed with the speedup.'

#    M = character_beam_estimator(llm, K, backend=1)
    M = character_beam_estimator(llm, K, backend=2)

#    if 1:
#    with timeit('run'):
    with profiler():
        for context in iterview(list(prefixes(qs))):
            M.logp_next(context)


def test_memory_benchmark():
    from arsenal.viz import update_ax

    llm = load_model_by_name('gpt2')
    K = 5

    context = """There are now rumblings that Apple might soon invade the smart watch space, \
though the company is maintaining its customary silence. The watch doesn't have a \
microphone or speaker, but you can use it to control the music on your phone. You can \
glance at the watch face to view the artist and title of a song."""

    T = timers()

    #C = character_beam_estimator(llm, K, backend=1)
    C = character_beam_estimator(llm, K, backend=2)

    ax = pl.figure().add_subplot(111)

    import gc
    gc.disable()

    surprisals = []
    for xs in iterview(list(prefixes(context))):
        if len(xs) == 0: continue
        with T['estimator'](N=len(xs)):

            before = time()
            logp = C.logp_next(xs)
            took = time() - before

        if len(xs) < len(context):
            x = context[len(xs)]
            print(f'{x!r} surprisal: {-logp[x]:.4f} {[len(xs), len(C.llm._cache), len(C.cache), len(C.beam.cache)]} {took:.4f} sec')
            surprisals.append(-logp[x])

#        gc.collect()

        with update_ax(ax):
            T.plot_feature('N', ax=ax, show_curve=(len(xs) > 2))
#        if len(xs) >= 142:
#            break

    pl.ioff()
    pl.show()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
