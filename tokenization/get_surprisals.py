import tqdm
from tokenization.lm import load_model_by_name
from tokenization.util import prefixes
from tokenization.character_beam import character_beam_estimator

def get_string_surprisals(string, estimator):
    
    surprisals = []
    for xxs in prefixes(string):
        try:
            logp = estimator.logp_next(xxs)
        except ValueError:
            print(f'"{xxs}"')
            raise ValueError
        if len(xxs) < len(string):
            x = string[len(xxs)]
            surprisals.append((x, -logp[x]))
        else:
            surprisals.append(("<EOS>", -logp[" "]))

    return surprisals



if __name__ == "__main__":
    for fname in ["celer", 'provo', 'ucl', 'mecoL1']:
        stimuli = list(open(f"{fname}.txt").readlines())
        model_name = 'gpt2'
        llm = load_model_by_name(model_name)


        if fname in ["ucl", "celer", "mecoL1", "provo", "test"]:
            print("running", fname)
            with open(f"{fname}_surprisals.{model_name}_K5.txt", "w") as fout:
                for x_id, x in tqdm.tqdm(enumerate(stimuli), total=len(stimuli)):
                    x = x.strip()
                    C = character_beam_estimator(llm, K=5)
                    surprisals = get_string_surprisals(x, C)
                    for token, surp in surprisals:
                        fout.write(f"{x_id} {token} {surp}\n")
                    fout.write("\n")
                    fout.flush()
                    C.llm.clear_cache()
