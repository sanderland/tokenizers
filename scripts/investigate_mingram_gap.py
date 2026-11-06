#!/usr/bin/env python3
"""Controlled reference MinGram ablations on the exact local Rosetta split."""
from __future__ import annotations
import contextlib, gzip, hashlib, json, math, tempfile
from collections import Counter
from pathlib import Path
from compare_mingram_compression import *
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array

class AblationTrainer(MinGramTrainer):
    def __init__(self, *args, protected=(), max_chars=None, smooth=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected=protected; self.max_chars=max_chars; self.smooth=smooth
    def _build_bpe_init_tokens(self):
        tokens=super()._build_bpe_init_tokens()
        if self.max_chars:
            tokens=[t for t in tokens if len(t.atomic_tokens)==1 or len(self.pretokenizer.decode(t.atomic_tokens, errors='replace'))<=self.max_chars]
        by_seq={tuple(t.atomic_tokens):t for t in tokens}
        for seq in self.protected:
            seq=tuple(seq)
            if seq in by_seq: by_seq[seq].required=True
            else:
                t=UnigramToken(id=max(t.id for t in tokens)+1,atomic_tokens=token_array(seq),log_prob=-100.,required=True)
                tokens.append(t);by_seq[seq]=t
        return tokens
    def run_m_step(self,model,counts):
        if not self.smooth: return super().run_m_step(model,counts)
        # Rust drops unused learned pieces, retaining required pieces at count 0.5.
        model=MinGramModel(model.pretokenizer,[t for t in model.tokens.values() if counts[t.id]>0 or t.required],score_delta=model.score_delta)
        total=sum(max(counts[t.id],.5) for t in model.tokens.values())
        for t in model.tokens.values(): t.log_prob=math.log(max(counts[t.id],.5)/total)
        return model,0

def main():
    output=ROOT/'docs/experiments/ctok-mingram-gap.json'
    source=ROOT.parent/'ctok/tests/fixtures/rosetta.jsonl.gz'
    holdout=source.with_name('rosetta_holdout.jsonl.gz')
    def texts(path):
        with gzip.open(path,'rt') as f:return [json.loads(line)['text'] for line in f]
    train,test=texts(source),texts(holdout)
    pt=UTF8Pretokenizer(UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX,remove_unassigned=False))
    char_set=set(''.join(pt.decode(c) for s in train for c in pt.pretokenize(s)))|{chr(i) for i in range(128)}|{chr(i) for i in range(0xFDD0,0xFDD5)}
    # Byte IDs are read from the pretokenizer, not assumed to start at zero.
    def atomic(raw):return [pt.byte_ids[b] for b in raw]
    protected=[atomic(c.encode()) for c in sorted(char_set)]
    prefixes=sorted({c.encode()[:n] for c in char_set for n in range(2,len(c.encode()))})
    protected += [atomic(p) for p in prefixes]
    variants=[('reference',{}),('charge_223_extra_slots',{'size':3873}),('max_16_characters',{'max_chars':16}),('rust_count_floor',{'smooth':True}),('protect_characters_and_prefixes',{'protected':protected}),('protected_plus_floor',{'protected':protected,'smooth':True}),('protected_plus_max16',{'protected':protected,'max_chars':16}),('protected_floor_max16',{'protected':protected,'smooth':True,'max_chars':16})]
    rows=[]
    with tempfile.TemporaryDirectory(prefix='gap-ablation-') as tmp,(Path(tmp)/'training.log').open('w') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):corpus=PretokenizedCorpus.from_texts('gap_rosetta',train,pt,base_path=tmp)
        for name,settings in variants:
            size=settings.pop('size',4096)
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                model=AblationTrainer(pt,corpus,MinGramTrainerConfig(additional_vocab_size=size-256,overshoot_factor=1.1,num_workers=1,verbose=False),**settings).train()
                path=str(Path(tmp)/f'{name}.json.gz');model.save(path);model=MinGramModel.load(path)
                count=sum(len(model.encode(s)) for s in test)
            row={'variant':name,'actual_size':len(model.tokens),'test_tokens':count,'gain_vs_bpe_pct':100*(1-count/79722)}
            rows.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256(holdout.read_bytes()).hexdigest(),'bpe_test_tokens':79722,'original_rust_test_tokens':79542,'required_characters':len(char_set),'prefixes':len(prefixes),'results':rows,'caveats':['Reference-only ablations; protective inventory retains all 256 atomic bytes (13 more than the Rust UTF-8 floor).','The 223-slot charge measures capacity sensitivity, not exact equivalence of vocabularies.','This fixture is not the paper corpus.']},indent=2)+'\n')
if __name__=='__main__':main()
