#!/usr/bin/env python3
"""Compare byte-alphabet MinGram with the existing Rosetta reference results."""
from collections import Counter
import gzip, hashlib, json
from compare_mingram_compression import ROOT, rust, UTF8Pretokenizer, UTF8PretokenizerConfig, GPT4O_REGEX

def main():
    source=ROOT.parent/'ctok/tests/fixtures/rosetta.jsonl.gz'
    holdout=source.with_name('rosetta_holdout.jsonl.gz')
    def texts(path):
        with gzip.open(path,'rt') as f:return [json.loads(line)['text'] for line in f]
    pt=UTF8Pretokenizer(UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX,remove_unassigned=False))
    # Same reversible byte alphabet as tk_encode::utils::byte_level.
    printable=list(range(33,127))+list(range(161,173))+list(range(174,256))
    mapping={b:chr(b) for b in printable}
    for index,b in enumerate(b for b in range(256) if b not in mapping): mapping[b]=chr(256+index)
    def mapped(raw):return ''.join(mapping[b] for b in raw)
    train=Counter(mapped(pt.token_to_bytes[a][0] for a in c) for s in texts(source) for c in pt.pretokenize(s))
    test=[mapped(pt.token_to_bytes[a][0] for a in c) for s in texts(holdout) for c in pt.pretokenize(s)]
    rows=[]
    for length in [16,64,256]:
        model=rust([{'op':'train','alphabet_mode':'byte_level','vocab_size':4096,'seed_factor':1.1,'max_length':length,'sentences':sorted(train.items())}])[0]
        if 'error' in model:raise RuntimeError(model)
        encoded=rust([{'op':'fixed','pieces':model['pieces'],'texts':test}])[0]
        if 'error' in encoded:raise RuntimeError(encoded)
        count=sum(map(len,encoded['encoded']))
        def safe(s):
            raw=bytes({c:b for b,c in mapping.items()}[c] for c in s)
            try:raw.decode();return True
            except UnicodeDecodeError as e:return e.start==0 and e.reason=='unexpected end of data'
        invalid=[s for s,_ in model['pieces'] if len(s)>1 and not safe(s)]
        assert not invalid,invalid[:5]
        row={'max_piece_length_bytes':length,'actual_size':model['size'],'test_tokens':count,'gain_vs_reference_bpe_pct':100*(1-count/79722),'invalid_multibyte_pieces':len(invalid)}
        rows.append(row);print(json.dumps(row),flush=True)
    output=ROOT/'docs/experiments/ctok-mingram-byte-alphabet.json'
    output.write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256(holdout.read_bytes()).hexdigest(),'reference_bpe_test_tokens':79722,'reference_mingram_test_tokens':79045,'old_rust_utf8_test_tokens':79542,'seed_factor':1.1,'vocab_size':4096,'results':rows,'caveats':['Same NFC and GPT4O regex chunks; Rust sees a reversible mapping of their UTF-8 bytes.','Rust BPE and script_tok BPE can choose different merges on equal frequencies.','Length counts byte atoms in this alphabet, not Unicode codepoints.','This fixture is not the paper corpus.']},indent=2)+'\n')
if __name__=='__main__':main()
