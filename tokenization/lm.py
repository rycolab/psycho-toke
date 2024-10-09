"""
Language models go here
"""

import numpy as np
import torch
import transformers

from collections import defaultdict
from arsenal.maths import sample_dict

from tokenization.util import unflatten, Chart


class LM:
    r"""We say that $p\colon V^* \to [0,1]$ is a language model if $p$ is a probability
    distribution over strings from some alphabet $V$ of tokens.

    Every language model admits a left-to-right factorization:

    $$
    p(x_1 x_2 \cdots x_T) = p(x_1 \mid \varepsilon) p(x_2 \mid x_1) \cdots p(x_T \mid x_1 \cdots x_{T-1}) p(\mathrm{EOS} \mid x_1 \cdots x_T)
    $$

    Arguments:

      - `V`: a vocabulary of symbols

      - `eos`: a distinguished end of sequence symbol

      - `p_next(xs)`: $p(\cdot \mid x_1 \cdots x_T)$ is provided by subclasses.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V

    def __call__(self, context):
        "Compute the probability of a complete string."
        # return np.exp(self.logp(ys))
        assert context[-1] == self.eos
        P = 1
        for i, y in enumerate(context):
            assert y in self.V, y
            p = self.p_next(context[:i])
            P *= p[y]
            if P == 0:
                break
        return P

    def logp(self, context):
        "Compute the probability of a complete string."
        assert context[-1] == self.eos
        return sum(self.logp_next(context[:i])[y] for i, y in enumerate(context))

    def logp_next(self, context):
        "Compute the log conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def p_next(self, context):
        "Compute the (conditional) distribution over the next token given the `prefix`."
        return self.p_next(context)

    def p_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        P = 1
        for i in range(len(extension)):
            p = self.p_next(context + extension[:i])
            P *= p[extension[i]]
        return P

    def logp_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        logP = 0
        for i in range(len(extension)):
            logp = self.logp_next(context + extension[:i])
            logP += logp[extension[i]]
        return logP

    def clear_cache(self):  # pragma: no cover
        pass

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        prob=True,
        verbose=0,
        max_tokens=np.inf,
        join=lambda ys, y: (ys, y),
    ):
        assert isinstance(ys, tuple), ys
        P = 1.0
        t = 0
        while True:
            p = self.p_next(ys).normalize()
            y = draw(p) if t <= max_tokens else self.eos
            P *= p[y]
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end='')
            if y == self.eos:
                return (ys, P) if prob else ys
            ys = join(ys, y)



from collections import OrderedDict

class TokenizedLLM(LM):
    """
    This is a simple class which wraps a token LLM with a tokenizer.
    """

    def __init__(self, tokenizer, model, batch_size, cache_size=128):
        self.tokenizer = tokenizer

        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)    # send the model to gpu, if one is available
        model.eval()             # Set the model in evaluation mode; avoids gradient overhead

        self._decode = decode_tokenizer_vocab(self.tokenizer)
        self._encode = {x: i for i, x in enumerate(self._decode)}

        self._cache = OrderedDict()
        self._cache_size = cache_size

        super().__init__(V=set(self._decode), eos=self.tokenizer.eos_token)

    def encode_prompt(self, prompt):
        "Encode `prompt` as a tuple of tokens (each a string)."
        return unflatten(tuple(self._decode[i] for i in self.tokenizer.encode(prompt)))

    def __call__(self, context):
        return self._model([self._encode[x] for x in context])

#    def __call__(self, context):
#        return np.exp(self.logp(context))

#    def logp(self, context):
#        input_ids = context
#        if isinstance(input_ids, list):
#            input_ids = torch.LongTensor([input_ids]).squeeze()
#        if input_ids[0] != self.model.config.bos_token_id:
#            input_ids = torch.cat(
#                [torch.LongTensor([self.model.config.bos_token_id]), input_ids]
#            )
#        with torch.no_grad():
#            input_ids = input_ids.to(self.device)
#            outputs = self.model(input_ids=input_ids, labels=input_ids)
#            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
#        token_lprobs = torch.gather(lprobs, 1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
#        return torch.sum(token_lprobs, dim=-1).item()

    def clear_cache(self):
        self._cache.clear()

    def get_state(self, context):
        assert isinstance(context, tuple) and (len(context) == 0 or len(context) == 2), context

        value = self._cache.get(context, None)
        if value is not None:
            self._cache.move_to_end(context)   # Move the key to the end to show it was recently used
            return value

        if len(context) == 0:
            # Note: initial state is uses BOS padding.
            input_ids = torch.LongTensor([self.tokenizer.bos_token_id]).to(self.device)
            value = self.model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=None,
                use_cache=True,
            )

        else:
            (xs, x) = context
            x = self._encode[x]
            prev_state = self.get_state(xs)
            input_ids = torch.LongTensor([x]).to(self.device)
            value = self.model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=prev_state.past_key_values,
                use_cache=True,
            )

        self._cache[context] = value
        if len(self._cache) > self._cache_size:
            # Pop the first item (the least recently used)
            self._cache.popitem(last=False)[0]

        return value

    def logp_next(self, context):
        return self.p_next(context, return_logp=True)

    def p_next(self, context, return_logp=False):
        assert isinstance(context, tuple) and len(context) == 0 or len(context) == 2, context
        #assert isinstance(context, tuple), 'API change; `context` must be explicitly tokenized'

        with torch.no_grad():
            outputs = self.get_state(context)
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        _logp = lprobs[0, :]

        if hasattr(_logp, 'cpu'):
            _logp = _logp.cpu().numpy()

        if return_logp:
            return LazyProb(_logp, self._encode, self._decode)
        else:
            _p = np.exp(_logp, dtype=np.float64)
            return LazyProb(_p, self._encode, self._decode)


class LazyProb:
    """
    This class is used to efficiently associate string with the indices of LLM's
    tokens distribution over next tokens.
    """

    def __init__(self, _p: torch.tensor, encode: dict[str, int], decode: list[str]):
        self._p = _p
        self._encode = encode
        self._decode = decode

    def normalize(self):
        return LazyProb(
            _p=self._p / self._p.sum(),
            encode=self._encode,
            decode=self._decode,
        )

    def keys(self):
        return self._decode

    def values(self):
        return self._p

    def items(self):
        return zip(self._decode, self._p)

    def __getitem__(self, token: str) -> float:
        i = self._encode.get(token)
        return self._p[i] if i is not None else 0

    def materialize(self, top=None):
        _p = self._p
        _decode = self._decode

        top_p = _p.argsort() if top is None else _p.argsort()[-int(top) :]

        pp = Chart(0)
        for i in reversed(top_p):
            pp[_decode[i]] = _p[i]

        return pp if top is None else pp.normalize()

    def __repr__(self):
        return repr(self.materialize())


def load_model_by_name(model_name, batch_size=None):
    """
    Load an LLM from 🤗 into a `TokenizedLLM`.
    """

    if model_name == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        return TokenizedLLM(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    if model_name == 'gpt2-large':
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name, use_fast=False)

        return TokenizedLLM(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    else:
        raise ValueError(model_name)


def decode_tokenizer_vocab(tokenizer):

    name = tokenizer.name_or_path.lower()
    if 'gpt2' in name:
        mapping = GPT2Mapping(tokenizer)
    elif 'llama-3' in name:
        mapping = LLaMaMapping(tokenizer)
    else:
        raise ValueError(f'We do not yet support tokenizer: {tokenizer.name_or_path}.')

    decoded = [mapping(i) for i in range(len(tokenizer))]

    check_collisions(decoded)

    return decoded


def check_collisions(decoded):
    # check for vocabulary collisions
    tmp = defaultdict(list)
    for i, t in enumerate(decoded):
        tmp[t].append(i)
    for x in tmp:
        assert len(tmp[x]) == 1, f'surface form {x!r} maps to more than one token> {tmp[x]}'


class Mapping:

    # Adapted from
    # https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py

    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    def _map(self, token_id: int) -> str:
        # if token_id is tensor, convert it to int
        if hasattr(token_id, 'item'):
            token_id = token_id.item()
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return raw_token

    def __call__(self, token_id: int) -> bytes:
        token = self._map(token_id)
        return bytes(token, 'utf-8')


class GPT2Mapping(Mapping):
    # Adapted from
    # https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py

    def __call__(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        if raw_token.startswith('Ġ'):
            raw_token = raw_token.replace('Ġ', ' ')
        if raw_token.startswith('Ċ'):
            raw_token = raw_token.replace('Ċ', '\n')
        if raw_token.startswith('ĉ'):
            raw_token = raw_token.replace('ĉ', '\t')
        return raw_token


class LLaMaMapping(Mapping):
    # Adapted from
    # https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py

    def __call__(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        raw_token = raw_token.replace('Ġ', ' ')
        raw_token = raw_token.replace('Ċ', '\n')
        raw_token = raw_token.replace('ĉ', '\t')
        raw_token = raw_token.replace('č', '\r')
        return raw_token
