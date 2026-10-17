"""Conservative lexical first-pass candidates, not a POS/semantic gold label."""
import csv,re,collections,json
from pathlib import Path
BASE=Path('data/clotho_aqa')
# Explicit source nouns keep adjective-only predicates out of nominal candidates.
NOUNS=set('dog cat bird animal person man woman baby child boy girl human people voice speech music motor engine machine car truck vehicle train plane airplane helicopter boat ship bus motorcycle bicycle bell alarm siren horn phone telephone clock water rain wind thunder storm thunderstorm waterfall river stream ocean sea waves fire footsteps noise sound static television radio guitar piano drum violin flute instrument chainsaw saw drill tool fan dryer vacuum rooster duck cow horse pig wolf lion owl cricket insect fly bee frog liquid fluid paper plastic glass metal wood gravel keyboard typing bubbling dripping splashing barking clapping coughing sneezing whistling humming buzzing ringing tapping knocking scraping rattling creaking clicking banging rustling traffic wind chimes echo crash scream splash breeze gear forklift'.split())
BAD=set('only one two three four five more less many multiple several single another other no not never same different loud quiet soft high low pitched repetitive constant continuous rhythmic steady fast slow big small large little far nearby distant close indoors outdoors inside outside background foreground distance end beginning time throughout entire whole always still alone happy angry clean dry wet sunny crowded natural artificial human-made man-made'.split())
VERB_BAD=set('being sounding getting becoming repeating occurring happening continuing increasing decreasing varying approaching receding trying doing making coming going moving standing sitting looking watching wearing'.split())
def noun_phrase(tokens):
 t=[x for x in tokens if x not in {'a','an','the','any','some','this','that'}]
 return bool(t) and len(t)<=4 and t[-1] in NOUNS and all(x in NOUNS or x in {'electric','electrical','musical','air','raid','rubber','running','flowing','flying','crinkling','cracking','crashing','popping','gurgling','squeaking'} for x in t) and not any(x in BAD or x.endswith('ly') for x in t) and not any(x in {'and','or','in','on','at','with','of','from','to','by','than'} for x in t)
def classify(q):
 t=re.findall(r"[a-z]+(?:'[a-z]+)?",q.lower())
 if not t or t.pop(0)!='is':return None
 if t and t[0]=='there':t=t[1:]
 elif t and t[0] in {'it','this','that'}:t=t[1:]
 if not t or any(x in BAD or x.endswith('ly') for x in t):return None
 # A single final present participle, with no trailing qualifiers/objects.
 if len(t)>=2 and t[-1].endswith('ing') and t[-1] not in VERB_BAD and noun_phrase(t[:-1]):
  return 'noun_ing',' '.join(t)
 if noun_phrase(t):return 'noun',' '.join(t)
 return None
summary={}
for split in ['train','test']:
 rows=list(csv.DictReader(open(BASE/f'clotho_aqa_{split}_is.csv')))
 grouped=collections.defaultdict(list)
 for r in rows:grouped[(r['file_name'],r['QuestionText'].strip())].append(r)
 keep=[]
 for (audio,q),rs in grouped.items():
  result=classify(q)
  if result:
   c=collections.Counter(r['answer'].strip().lower() for r in rs)
   best=c.most_common();answer=best[0][0] if len(best)==1 or best[0][1]>best[1][1] else 'tie'
   keep.append(dict(file_name=audio,QuestionText=q,form=result[0],target_candidate=result[1],answer=answer,yes_votes=c['yes'],no_votes=c['no'],annotation_count=len(rs)))
 path=BASE/f'clotho_aqa_{split}_is_noun_candidates.csv'
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=['file_name','QuestionText','form','target_candidate','answer','yes_votes','no_votes','annotation_count']);w.writeheader();w.writerows(keep)
 summary[split]={'questions':len(keep),'audio':len(set(r['file_name'] for r in keep)),'forms':dict(collections.Counter(r['form'] for r in keep)),'answers':dict(collections.Counter(r['answer'] for r in keep))}
 print(split,summary[split])
 for form in ['noun_ing','noun']:
  print(form,list(dict.fromkeys(r['QuestionText'] for r in keep if r['form']==form))[:12])
(BASE/'noun_candidates_summary.json').write_text(json.dumps(summary,indent=2))
