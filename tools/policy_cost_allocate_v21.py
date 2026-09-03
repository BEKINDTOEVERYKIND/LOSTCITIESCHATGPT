#!/usr/bin/env python3
"""Outcome-blind frozen allocation for policy-frequency-cost evidence."""
from __future__ import annotations

import argparse, hashlib, json, os, re, tempfile
from pathlib import Path
from typing import Any, Iterable

TRAIN_SCHEMA="LCPOLICYCOST-TRAIN-RESERVOIR-V5"
VECTOR_SCHEMA="LCPOLICYCOST-VECTOR-RESERVOIR-V2"
TRAIN_MANIFEST="LCPOLICYCOST-TRAIN-ALLOCATION-V5"
VECTOR_MANIFEST="LCPOLICYCOST-VECTOR-ALLOCATION-V2"
TRAIN_QUOTA=16; VECTOR_QUOTA=64
TRAIN_CAP=1024; VECTOR_CAP=64
SOURCE_STATE_LIMIT=900
MASK_MAX=5; UNION_MAX=5
ROUNDS=range(3); PLY_BINS=range(24); RATIO_BINS=range(6); PAIR_TYPES=range(2)
SLOTS=range(MASK_MAX); VECTOR_GROUPS=range(3); VECTOR_GROUP_QUOTAS=(22,21,21)
SEEDS={"TRAIN":202808100101,"SELECT":202808100201,"TEST":202808100301}
MATCHES={"TRAIN":65536,"SELECT":32768,"TEST":32768}
PLY_BOUNDARIES=(0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,48,64)
PLY_INTERVALS=[[PLY_BOUNDARIES[i],PLY_BOUNDARIES[i+1]] for i in range(24)]
FLOOR_BITS=["3c23d70a","3ca3d70a"]
TRAIN_RULE=(b"lc-policy-cost-train-allocation-v5|canonical-greedy-selection|"
 b"global-source-unique|quota16|rank-major-diagonal-cell-interleave-v1")
VECTOR_RULE=(b"lc-policy-cost-vector-allocation-v2|64-fixed-g0-g1-tail|"
 b"priority-v1|rank-major-three-band-base-interleave-v1")
HEX64=re.compile(r"[0-9a-f]{64}\Z"); TOKEN=re.compile(r"[A-Za-z0-9_.:-]+\Z")
TRAIN_COLUMNS=("cell","priority_sha256","source_match_index","source_state_index",
 "source_match_id","round","ply_bin","ratio_bin","pair_type","pair_move_a",
 "pair_move_b","orbit_sha256","state_sha256","mask_001_sha256",
 "mask_002_sha256","master_sha256","state_hex")
VECTOR_COLUMNS=("cell","priority_sha256","source_match_index","source_state_index",
 "source_match_id","round","ply_bin","frontier_present","allocation_slot",
 "master_width","orbit_sha256","state_sha256","mask_001_sha256",
 "mask_002_sha256","master_sha256","state_hex")

class AllocationError(ValueError): pass

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  while b:=f.read(1<<20): h.update(b)
 return h.hexdigest()

def checked(path:Path,expected:str,label:str)->None:
 if HEX64.fullmatch(expected) is None or sha256(path)!=expected: raise AllocationError(f"{label} SHA-256 mismatch")

def unique(pairs:list[tuple[str,Any]])->dict[str,Any]:
 d={}
 for k,v in pairs:
  if k in d: raise AllocationError(f"duplicate JSON key {k}")
  d[k]=v
 return d

def discovery(path:Path,expected:str)->tuple[dict[str,Any],dict[str,Any],dict[str,Any]]:
 checked(path,expected,"discovery"); raw=path.read_bytes()
 if not raw.endswith(b"\n") or b"\r" in raw: raise AllocationError("discovery is not canonical LF")
 try:
  rows=[json.loads(x,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(AllocationError(x))) for x in raw.decode("ascii").splitlines()]
 except (UnicodeError,json.JSONDecodeError) as e: raise AllocationError(f"invalid discovery: {e}") from e
 if len(rows)!=3 or [x.get("record_type") for x in rows] != ["header","census","footer"] or any(x.get("schema")!="lc-policy-cost-discovery-v5" for x in rows): raise AllocationError("discovery schema mismatch")
 return rows[0],rows[1],rows[2]

def json_uint(value:Any,label:str)->int:
 if isinstance(value,bool) or not isinstance(value,int) or value<0:
  raise AllocationError(f"invalid discovery {label}")
 return value

def campaign_discovery_header(split:str,rh:dict[str,Any])->dict[str,Any]:
 return {
  "schema":"lc-policy-cost-discovery-v5","record_type":"header","split":split,
  "purpose":"locked_campaign","seed":str(SEEDS[split]),
  "seed_domain":"locked-discovery","match_start":0,
  "requested_matches":MATCHES[split],"generator":"policy20_self_play",
  "state_import_supported":False,"symmetries":20,"net_path":"data/champion.bin",
  "net_sha256":rh["net"],"exclusion_manifest_sha256":rh["exclusion"],
  "exclusion_orbits":17,"floor_bits":FLOOR_BITS,
  "shortlist":{"root_width":5,"action_core_count":3,"min_candidates":1,
               "candidate_mass":0},
  "master_max":MASK_MAX,"truth_support_max":6,
  "reservoir_method":"bounded_sha256_priority",
  "reservoir_per_subcell":TRAIN_CAP if split=="TRAIN" else VECTOR_CAP,
  "ply_bins":PLY_INTERVALS,"pooled_ge64":"census_only",
  "vector_slot":"sha256(split_seed,state_sha256) mod master_width",
  "vector_poststratum":"j0,j1,j>=2",
  "frontier":"1pct admits aggregate semantic core in [0.01,0.02) removed by 2pct",
  "burned_source_deal_seeds":(
   "1..200, maintained-800 seed 1, 202611010101, all policy-cost-v1 "
   "fixed seeds in 20261110/11/12/13/14/15/16/21/22, every 20261129 "
   "feasibility-smoke seed, all policy-cost-v2 fixed seeds in "
   "20261210/11/12/13/14/15/16/21/22, every 20261229 feasibility-smoke "
   "seed, 202612010101, all policy-cost-v3 fixed seeds in "
   "20270110/11/12/13/14/15/16/21/22, every 20270129 "
   "feasibility-smoke seed, 202701010101, all policy-cost-v4 fixed "
   "seeds in 20270210/11/12/13/14/15/16/21/22, every 20270229 "
   "feasibility-smoke seed, 202702010101, all policy-cost-v5 fixed "
   "seeds in 20270310/11/12/13/14/15/16/21/22, every 20270329 "
   "feasibility-smoke seed, 202703010101, all policy-cost-v6 fixed "
   "seeds in 20270410/11/12/13/14/15/16/21/22, every 20270429 "
   "feasibility-smoke seed, 202704010101, all policy-cost-v7 fixed "
   "seeds in 20270510/11/12/13/14/15/16/21/22, every 20270529 "
   "feasibility-smoke seed, 202705010101, all policy-cost-v8 fixed "
   "seeds in 20270710/11/12/13/14/15/16/21/22, every 20270729 "
   "feasibility-smoke seed, 202707010101, all policy-cost-v9 fixed "
   "seeds in 20270810/11/12/13/14/15/16/21/22, every 20270829 "
   "feasibility-smoke seed, 202708010101, all policy-cost-v10 fixed "
   "seeds in 20270910/11/12/13/14/15/16/21/22, every 20270929 "
   "feasibility-smoke seed, 202709010101, all policy-cost-v11 fixed "
   "seeds in 20271010/11/12/13/14/15/16/21/22, every 20271029 "
   "feasibility-smoke seed, 202710010101, all policy-cost-v12 fixed "
   "seeds in 20271110/11/12/13/14/15/16/21/22, every 20271129 "
   "feasibility-smoke seed, 202711010101, all policy-cost-v14 fixed seeds in 20280110/11/12/13/14/15/16/21/22, every 20280129 feasibility-smoke seed, 202801010101, 202802010101, all policy-cost-v15 fixed seeds in 20280210/11/12/13/14/15/16/21/22, every 20280229 feasibility-smoke seed, 202803010101, all policy-cost-v16 fixed seeds in 20280310/11/12/13/14/15/16/21/22, every 20280329 feasibility-smoke seed, 202804010101, all policy-cost-v17 fixed seeds in 20280410/11/12/13/14/15/16/21/22, every 20280429 feasibility-smoke seed, 202805010101, all policy-cost-v18 fixed seeds in 20280510/11/12/13/14/15/16/21/22, every 20280529 feasibility-smoke seed, 202806010101, all policy-cost-v19 fixed seeds in 20280610/11/12/13/14/15/16/21/22, every 20280629 feasibility-smoke seed, 202807010101, all policy-cost-v20 fixed seeds in 20280710/11/12/13/14/15/16/21/22, every 20280729 feasibility-smoke seed, 202808010101, and every "
   "20280829 feasibility-smoke seed"
  ),
 }

def validate_campaign_discovery(dh:dict[str,Any],dc:dict[str,Any],df:dict[str,Any],
                                rh:dict[str,Any],rf:dict[str,Any],split:str)->None:
 """Bind a complete, exact, cap-free discovery census before allocation."""
 expected_header=campaign_discovery_header(split,rh)
 if dh!=expected_header: raise AllocationError("locked discovery header mismatch")
 if set(df)!={"schema","record_type","requested_matches","completed_matches",
             "attempted_states","accepted_states","probe_orbit_rejections",
             "cap_hits","eligible_units","retained_units",
             "units_rejected_by_bound"} or \
    df.get("schema")!="lc-policy-cost-discovery-v5" or \
    df.get("record_type")!="footer":
  raise AllocationError("locked discovery footer mismatch")
 values={key:json_uint(df.get(key),key) for key in (
  "requested_matches","completed_matches","attempted_states","accepted_states",
  "probe_orbit_rejections","cap_hits","eligible_units","retained_units",
  "units_rejected_by_bound")}
 if values["requested_matches"]!=MATCHES[split] or \
    values["completed_matches"]!=MATCHES[split]:
  raise AllocationError("discovery match census incomplete")
 if values["cap_hits"]!=0:
  raise AllocationError("discovery cap hit")
 if values["attempted_states"] != \
    values["accepted_states"]+values["probe_orbit_rejections"]:
  raise AllocationError("discovery state census algebra mismatch")
 if (values["eligible_units"],values["retained_units"],
     values["units_rejected_by_bound"]) != \
    (rf["eligible"],rf["retained"],rf["rejected"]):
  raise AllocationError("discovery/reservoir footer count mismatch")

 required_census={"schema","record_type","state_commitment_chain_sha256",
                  "accepted_by_round","pooled_ge64_by_round",
                  "exact_terminal_preempted_by_round","mask_width_counts",
                  "union_width_counts","eligible_master_width_counts",
                  "allocation_cells"}
 if set(dc)!=required_census or dc.get("schema")!="lc-policy-cost-discovery-v5" or \
    dc.get("record_type")!="census" or \
    dc.get("state_commitment_chain_sha256")!=rf["chain"]:
  raise AllocationError("locked discovery census mismatch")
 def uint_vector(value:Any,width:int,label:str)->list[int]:
  if not isinstance(value,list) or len(value)!=width:
   raise AllocationError(f"invalid discovery {label}")
  return [json_uint(item,label) for item in value]
 accepted_by_round=uint_vector(dc.get("accepted_by_round"),3,"accepted_by_round")
 pooled=uint_vector(dc.get("pooled_ge64_by_round"),3,"pooled_ge64_by_round")
 terminal=uint_vector(dc.get("exact_terminal_preempted_by_round"),3,
                      "exact_terminal_preempted_by_round")
 masks=dc.get("mask_width_counts")
 if not isinstance(masks,list) or len(masks)!=2:
  raise AllocationError("invalid discovery mask_width_counts")
 mask_rows=[uint_vector(row,MASK_MAX,"mask_width_counts") for row in masks]
 unions=uint_vector(dc.get("union_width_counts"),UNION_MAX,
                    "union_width_counts")
 if unions != mask_rows[0]:
  raise AllocationError(
   "union width histogram does not equal the 1pct master histogram")
 master_widths=uint_vector(dc.get("eligible_master_width_counts"),MASK_MAX,
                           "eligible_master_width_counts")
 accepted=values["accepted_states"]
 if sum(accepted_by_round)!=accepted or any(sum(row)!=accepted for row in mask_rows) or \
    sum(unions)!=accepted:
  raise AllocationError("discovery accepted-state census mismatch")
 if sum(pooled)!=rf["pooled"] or any(pooled[i]+terminal[i]>accepted_by_round[i]
                                     for i in range(3)):
  raise AllocationError("discovery terminal/tail census mismatch")
 if split=="TRAIN" and any(master_widths):
  raise AllocationError("TRAIN vector width census must be zero")
 if split!="TRAIN" and (sum(master_widths)!=rf["eligible"] or
                         rf["eligible"]+sum(pooled)+sum(terminal)!=accepted):
  raise AllocationError("vector width/state partition census mismatch")

def hv(lines:list[str],i:int,key:str)->str:
 x=lines[i].split("\t")
 if len(x)!=2 or x[0]!=key or not x[1]: raise AllocationError(f"expected header {key}")
 return x[1]

def uint(x:str,label:str)->int:
 if not x.isascii() or not x.isdigit(): raise AllocationError(f"invalid {label}")
 return int(x)

def train_priority(seed:int,r:dict[str,str])->str:
 b=seed.to_bytes(8,"little")+uint(r["source_match_index"],"source").to_bytes(8,"little")+uint(r["source_state_index"],"state").to_bytes(4,"little")
 b+=bytes((uint(r["round"],"round"),uint(r["ply_bin"],"ply"),uint(r["ratio_bin"],"ratio"),uint(r["pair_type"],"type")))
 b+=uint(r["pair_move_a"],"move").to_bytes(2,"little")+uint(r["pair_move_b"],"move").to_bytes(2,"little")
 return hashlib.sha256(b"lc-policy-cost-reservoir-priority-v1"+b+bytes.fromhex(r["state_sha256"])).hexdigest()

def vector_slot(seed:int,state_hash:str,width:int)->int:
 d=hashlib.sha256(b"lc-policy-cost-vector-slot-v1"+seed.to_bytes(8,"little")+bytes.fromhex(state_hash)).digest()
 return int.from_bytes(d[:8],"little")%width

def vector_group(slot:int)->int:
 return min(slot,2)

def vector_priority(seed:int,r:dict[str,str])->str:
 b=seed.to_bytes(8,"little")+uint(r["source_match_index"],"source").to_bytes(8,"little")+uint(r["source_state_index"],"state").to_bytes(4,"little")
 b+=bytes((uint(r["round"],"round"),uint(r["ply_bin"],"ply"),uint(r["frontier_present"],"frontier"),uint(r["allocation_slot"],"slot")))
 return hashlib.sha256(b"lc-policy-cost-vector-priority-v1"+b+bytes.fromhex(r["state_sha256"])).hexdigest()

def reservoir(path:Path,expected:str,*,smoke_split:str|None=None,
              smoke_seed:int|None=None,smoke_matches:int|None=None,
              smoke_cap:int|None=None
              )->tuple[dict[str,Any],list[dict[str,str]],dict[str,Any]]:
 checked(path,expected,"reservoir"); raw=path.read_bytes()
 if not raw.endswith(b"\n") or b"\r" in raw: raise AllocationError("reservoir is not canonical LF")
 try: lines=raw.decode("ascii").splitlines()
 except UnicodeError as e: raise AllocationError("reservoir is not ASCII") from e
 if len(lines)<9 or lines[0] not in (TRAIN_SCHEMA,VECTOR_SCHEMA): raise AllocationError("reservoir schema mismatch")
 split=hv(lines,1,"split"); head={"schema":lines[0],"split":split,"purpose":hv(lines,2,"purpose"),"seed":uint(hv(lines,3,"seed"),"seed"),"net":hv(lines,4,"net_sha256"),"exclusion":hv(lines,5,"exclusion_sha256"),"cap":uint(hv(lines,6,"reservoir_per_subcell"),"cap")}
 smoke=smoke_split is not None
 if split not in SEEDS or (split=="TRAIN")!=(lines[0]==TRAIN_SCHEMA):
  raise AllocationError("reservoir campaign identity mismatch")
 if smoke:
  if smoke_seed is None or smoke_matches is None or smoke_cap is None or \
     split!=smoke_split or head["purpose"]!="smoke" or \
     head["seed"]!=smoke_seed or head["cap"]!=smoke_cap:
   raise AllocationError("reservoir smoke identity mismatch")
  source_match_limit=smoke_matches
 else:
  if head["purpose"]!="campaign" or head["seed"]!=SEEDS[split]:
   raise AllocationError("reservoir campaign identity mismatch")
  if head["cap"]!=(TRAIN_CAP if split=="TRAIN" else VECTOR_CAP):
   raise AllocationError("reservoir cap changed")
  source_match_limit=MATCHES[split]
 if HEX64.fullmatch(head["net"]) is None or HEX64.fullmatch(head["exclusion"]) is None: raise AllocationError("bad header hash")
 cols=TRAIN_COLUMNS if split=="TRAIN" else VECTOR_COLUMNS
 if lines[7].split("\t") != ["columns",*cols]: raise AllocationError("reservoir columns mismatch")
 rows=[]; previous={}; counts={}; seen_states=set(); footer=None
 for line in lines[8:]:
  if line.startswith("footer\t"):
   if footer is not None: raise AllocationError("duplicate footer")
   footer=line; continue
  if footer is not None: raise AllocationError("row after footer")
  v=line.split("\t")
  if len(v)!=len(cols): raise AllocationError("row width mismatch")
  r=dict(zip(cols,v,strict=True)); cell=r["cell"]
  if TOKEN.fullmatch(cell) is None or any(HEX64.fullmatch(r[k]) is None for k in ("priority_sha256","orbit_sha256","state_sha256","mask_001_sha256","mask_002_sha256","master_sha256")): raise AllocationError("invalid row token/hash")
  try: state=bytes.fromhex(r["state_hex"])
  except ValueError as e: raise AllocationError("bad state hex") from e
  if len(state)!=174 or state[0]!=1 or state[165] not in (0,1) or state[166]!=0:
   raise AllocationError("state is not the exact native information-view encoding")
  if hashlib.sha256(state).hexdigest()!=r["state_sha256"]: raise AllocationError("state hash mismatch")
  source=uint(r["source_match_index"],"source")
  state_index=uint(r["source_state_index"],"state")
  rd=uint(r["round"],"round"); pb=uint(r["ply_bin"],"ply")
  if source>=source_match_limit: raise AllocationError("source match is outside frozen discovery")
  if state_index>=SOURCE_STATE_LIMIT:
   raise AllocationError("source state is outside native match bound")
  if r["source_match_id"]!=f"{split}-{source:012d}" or rd not in ROUNDS or pb not in PLY_BINS: raise AllocationError("source/stratum mismatch")
  if split=="TRAIN":
   rb=uint(r["ratio_bin"],"ratio"); tp=uint(r["pair_type"],"type")
   if rb not in RATIO_BINS or tp not in PAIR_TYPES or cell!=f"r{rd}.p{pb}.g{rb}.t{tp}" or train_priority(int(head["seed"]),r)!=r["priority_sha256"]: raise AllocationError("TRAIN cell/priority mismatch")
  else:
   fr=uint(r["frontier_present"],"frontier"); sl=uint(r["allocation_slot"],"slot"); width=uint(r["master_width"],"width")
   key=(r["source_match_index"],r["source_state_index"])
   if fr not in (0,1) or not 1<=width<=MASK_MAX or sl>=width or cell!=f"r{rd}:p{pb:02d}:f{fr}:g{vector_group(sl)}" or r["master_sha256"]!=r["mask_001_sha256"] or vector_slot(int(head["seed"]),r["state_sha256"],width)!=sl or vector_priority(int(head["seed"]),r)!=r["priority_sha256"] or key in seen_states: raise AllocationError("vector cell/slot/priority mismatch")
   seen_states.add(key)
  if previous.get(cell,"")>=r["priority_sha256"]: raise AllocationError("priorities not strictly ordered")
  previous[cell]=r["priority_sha256"]; counts[cell]=counts.get(cell,0)+1
  if counts[cell]>head["cap"]: raise AllocationError("subcell exceeds cap")
  rows.append(r)
 if footer is None: raise AllocationError("missing footer")
 f=footer.split("\t")
 if len(f)!=11 or [f[i] for i in (0,1,3,5,7,9)] != ["footer","eligible_units","retained_units","rejected_by_bound","state_commitment_chain_sha256","pooled_ge64_observed"]: raise AllocationError("footer mismatch")
 foot={"eligible":uint(f[2],"eligible"),"retained":uint(f[4],"retained"),"rejected":uint(f[6],"rejected"),"chain":f[8],"pooled":uint(f[10],"pooled"),"counts":counts}
 if foot["retained"]!=len(rows) or foot["eligible"]!=foot["retained"]+foot["rejected"] or HEX64.fullmatch(foot["chain"]) is None: raise AllocationError("footer counts mismatch")
 return head,rows,foot

def validate_contract_smoke(dp:Path,dhx:str,rp:Path,rhx:str,*,split:str,
                            seed:int,matches:int,reservoir_per_cell:int)->str:
 """Validate a real, small native v21 producer census with no efficacy path.

 The smoke identity is permanently outside every campaign domain.  Its sole
 purpose is to exercise the exact native producer/Python consumer boundary
 before a definition can launch the fixed production roots.
 """
 if split not in SEEDS:
  raise AllocationError("invalid contract-smoke split")
 seed_text=str(seed)
 if len(seed_text)!=12 or not seed_text.startswith("20280829"):
  raise AllocationError("contract smoke seed is outside burned 20280829")
 if not 1<=matches<=16:
  raise AllocationError("contract smoke matches must be in 1..16")
 max_cap=TRAIN_CAP if split=="TRAIN" else VECTOR_CAP
 if not 1<=reservoir_per_cell<=max_cap:
  raise AllocationError("invalid contract smoke reservoir cap")
 dh,dc,df=discovery(dp,dhx)
 rh,_,rf=reservoir(
  rp,rhx,smoke_split=split,smoke_seed=seed,smoke_matches=matches,
  smoke_cap=reservoir_per_cell)
 expected_header=campaign_discovery_header(split,rh)
 expected_header.update({
  "purpose":"feasibility_smoke_excluded_from_campaign",
  "seed":seed_text,"seed_domain":"20280829-smoke",
  "requested_matches":matches,
  "reservoir_per_subcell":reservoir_per_cell,
 })
 if dh!=expected_header:
  raise AllocationError("native contract-smoke header mismatch")
 if df.get("requested_matches")!=matches or \
    df.get("completed_matches")!=matches:
  raise AllocationError("native contract-smoke match census incomplete")
 # Reuse the full production census validator after substituting only the
 # deliberately different smoke identity and size.  All producer/consumer
 # schemas, algebra, five-bin widths, partitions, and reservoir counts remain
 # the actual native output.
 campaign_footer=dict(df)
 campaign_footer["requested_matches"]=MATCHES[split]
 campaign_footer["completed_matches"]=MATCHES[split]
 validate_campaign_discovery(
  campaign_discovery_header(split,rh),dc,campaign_footer,rh,rf,split)
 payload=(
  b"lc-policy-cost-v21-native-consumer-contract-v1\0"+
  bytes.fromhex(dhx)+bytes.fromhex(rhx)+split.encode("ascii")+b"\0"+
  seed.to_bytes(8,"little")+matches.to_bytes(4,"little")+
  reservoir_per_cell.to_bytes(4,"little")
 )
 return hashlib.sha256(payload).hexdigest()

def train_cells()->Iterable[str]:
 for r in ROUNDS:
  for p in PLY_BINS:
   for g in RATIO_BINS:
    for t in PAIR_TYPES: yield f"r{r}.p{p}.g{g}.t{t}"

def vector_bases()->Iterable[tuple[int,int,int]]:
 for r in ROUNDS:
  for p in PLY_BINS:
   for f in (0,1): yield r,p,f

def train_schedule_cells()->Iterable[str]:
 """Outcome-blind order spreading every quota rank over all fixed factors."""
 for base_type in PAIR_TYPES:
  for base_ratio in RATIO_BINS:
   for rd in ROUNDS:
    for pb in PLY_BINS:
     ratio=(base_ratio+pb)%len(RATIO_BINS)
     pair_type=(base_type+rd+pb)%len(PAIR_TYPES)
     yield f"r{rd}.p{pb}.g{ratio}.t{pair_type}"

def vector_schedule_bases()->Iterable[tuple[int,int,int]]:
 """Spread each 48-row evaluator slice over rounds/frontiers/ply bands."""
 for low_ply in range(8):
  for band in range(3):
   pb=low_ply+8*band
   for frontier in (0,1):
    for rd in ROUNDS:
     yield rd,pb,frontier

def allocate_train(rows:list[dict[str,str]])->list[dict[str,str]]:
 by={c:[] for c in train_cells()}
 for r in rows:
  if r["cell"] not in by: raise AllocationError("unknown TRAIN cell")
  by[r["cell"]].append(r)
 chosen_by={}; used=set()
 for cell in train_cells():
  chosen=[]
  for r in by[cell]:
   if r["source_match_index"] in used: continue
   used.add(r["source_match_index"]); chosen.append(r)
   if len(chosen)==TRAIN_QUOTA: break
  if len(chosen)!=TRAIN_QUOTA: raise AllocationError(f"sparse TRAIN cell {cell}; no top-up")
  chosen_by[cell]=chosen
 schedule=list(train_schedule_cells())
 if len(schedule)!=len(chosen_by) or set(schedule)!=set(chosen_by):
  raise AllocationError("TRAIN scheduling permutation drift")
 return [chosen_by[cell][rank] for rank in range(TRAIN_QUOTA)
         for cell in schedule]

def allocate_vectors(rows:list[dict[str,str]],N:dict[str,int])->tuple[list[dict[str,str]],dict[str,int]]:
 by={}
 for r in rows: by.setdefault(r["cell"],[]).append(r)
 selected_by_base={}; q={f"r{rd}:p{pb:02d}:f{fr}:g{g}":VECTOR_GROUP_QUOTAS[g]
                       for rd,pb,fr in vector_bases() for g in VECTOR_GROUPS}
 for rd,pb,fr in vector_bases():
  chosen=[]
  for g in VECTOR_GROUPS:
   cell=f"r{rd}:p{pb:02d}:f{fr}:g{g}"
   if N[cell]<q[cell] or len(by.get(cell,[]))<q[cell]:
    raise AllocationError(f"sparse fixed vector poststratum {cell}")
   slot_rows=by[cell][:q[cell]]
   if len({x["source_match_index"] for x in slot_rows})<8:
    raise AllocationError(f"vector poststratum {cell} lacks 8 sources")
   chosen+=slot_rows
  if len(chosen)!=VECTOR_QUOTA: raise AllocationError(f"base r{rd}.p{pb}.f{fr} lacks 64 vectors")
  selected_by_base[(rd,pb,fr)]=chosen
 schedule=list(vector_schedule_bases())
 if len(schedule)!=len(selected_by_base) or set(schedule)!=set(selected_by_base):
  raise AllocationError("vector scheduling permutation drift")
 selected=[selected_by_base[base][rank] for rank in range(VECTOR_QUOTA)
           for base in schedule]
 ids=[(x["source_match_index"],x["source_state_index"]) for x in selected]
 if len(ids)!=len(set(ids)): raise AllocationError("selected vector state reused")
 return selected,q

def atomic(path:Path,data:bytes)->None:
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists(): raise AllocationError(f"refusing to overwrite {path}")
 fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.partial.",dir=path.parent)
 try:
  with os.fdopen(fd,"wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  try: os.unlink(tmp)
  except FileNotFoundError: pass

def build_manifest(dp:Path,dhx:str,rp:Path,rhx:str)->bytes:
 dh,dc,df=discovery(dp,dhx); rh,rows,rf=reservoir(rp,rhx); split=rh["split"]
 validate_campaign_discovery(dh,dc,df,rh,rf,split)
 pooled=dc.get("pooled_ge64_by_round")
 if not isinstance(pooled,list) or len(pooled)!=3 or sum(pooled)!=rf["pooled"]: raise AllocationError("tail census mismatch")
 if (df.get("eligible_units"),df.get("retained_units"),df.get("units_rejected_by_bound"))!=(rf["eligible"],rf["retained"],rf["rejected"]): raise AllocationError("unit count mismatch")
 cells=dc.get("allocation_cells")
 if not isinstance(cells,list): raise AllocationError("allocation census absent")
 N={}; R={}; widths={}
 for x in cells:
  expected_keys=({"cell","eligible_units","retained_units"} if split=="TRAIN" else
                 {"cell","eligible_vectors","retained_vectors",
                  "master_width_histogram"})
  if not isinstance(x,dict) or set(x)!=expected_keys or not isinstance(x.get("cell"),str):
   raise AllocationError("bad census cell")
  ek="eligible_units" if split=="TRAIN" else "eligible_vectors"
  rk="retained_units" if split=="TRAIN" else "retained_vectors"; cell=x["cell"]
  if cell in N: raise AllocationError("duplicate census cell")
  N[cell]=json_uint(x[ek],f"{cell} eligible")
  R[cell]=json_uint(x[rk],f"{cell} retained")
  if split!="TRAIN":
   hist=x.get("master_width_histogram")
   if not isinstance(hist,list) or len(hist)!=MASK_MAX or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in hist) or sum(hist)!=N[cell]: raise AllocationError("bad per-cell width histogram")
   widths[cell]=hist
 expected_cells=(set(train_cells()) if split=="TRAIN" else
                 {f"r{rd}:p{pb:02d}:f{fr}:g{g}"
                  for rd,pb,fr in vector_bases() for g in VECTOR_GROUPS})
 if set(N)!=expected_cells: raise AllocationError("allocation census cell set mismatch")
 if any(N[c]<R[c] for c in N): raise AllocationError("retained count exceeds eligible census")
 if any(R[c]!=rf["counts"].get(c,0) for c in R): raise AllocationError("census/reservoir cell mismatch")
 if sum(N.values())!=rf["eligible"] or sum(R.values())!=rf["retained"]:
  raise AllocationError("aggregate census/reservoir count mismatch")
 common=[f"discovery_sha256\t{dhx}",f"reservoir_sha256\t{rhx}",
         f"source_net_sha256\t{rh['net']}",
         f"source_exclusion_sha256\t{rh['exclusion']}"]
 if split=="TRAIN":
  sel=allocate_train(rows)
  rule=hashlib.sha256(TRAIN_RULE).hexdigest()
  lines=[TRAIN_MANIFEST,"split\tTRAIN","purpose\tcampaign",*common,f"eligible_pair_commitment_sha256\t{rf['chain']}",f"allocation_rule_sha256\t{rule}",f"quota_per_cell\t{TRAIN_QUOTA}",f"eligible_units\t{rf['eligible']}",f"retained_reservoir_units\t{rf['retained']}",f"probe_orbit_rejections\t{df['probe_orbit_rejections']}",f"pooled_ge64_observed\t{rf['pooled']}",f"records\t{len(sel)}","columns\tallocation_id\tsource_match_index\tsource_state_index\tsource_match_id\tstate_id\tpair_id\tcell\tround\tply_bin\tratio_bin\tpair_type\tpair_move_a\tpair_move_b\torbit_sha256\tstate_sha256\tpair_sha256\tallocation_priority_sha256\tmask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex"]
  for i,r in enumerate(sel):
   state_id=f"{r['source_match_id']}:s{int(r['source_state_index']):03d}"
   pair_id=f"{int(r['pair_move_a']):05d}-{int(r['pair_move_b']):05d}"
   pair_sha=hashlib.sha256(bytes.fromhex(r["state_sha256"])+int(r["pair_move_a"]).to_bytes(2,"little")+int(r["pair_move_b"]).to_bytes(2,"little")).hexdigest()
   lines.append("\t".join((str(i),r["source_match_index"],r["source_state_index"],
    r["source_match_id"],state_id,pair_id,r["cell"],r["round"],r["ply_bin"],r["ratio_bin"],
    r["pair_type"],r["pair_move_a"],r["pair_move_b"],r["orbit_sha256"],
    r["state_sha256"],pair_sha,r["priority_sha256"],r["mask_001_sha256"],r["mask_002_sha256"],
    r["master_sha256"],r["state_hex"])))
 else:
  sel,q=allocate_vectors(rows,N); total=sum(N.values())
  if total!=rf["eligible"]: raise AllocationError("vector census total mismatch")
  aggregate=[sum(widths[c][w] for c in widths) for w in range(MASK_MAX)]
  if dc.get("eligible_master_width_counts")!=aggregate: raise AllocationError("aggregate width histogram mismatch")
  rule=hashlib.sha256(VECTOR_RULE).hexdigest()
  lines=[VECTOR_MANIFEST,f"split\t{split}","purpose\tcampaign",*common,f"eligible_state_commitment_sha256\t{rf['chain']}",f"allocation_rule_sha256\t{rule}",f"quota_per_base_cell\t{VECTOR_QUOTA}","source_minimum_per_poststratum\t8",f"total_census\t{total}",f"retained_reservoir_vectors\t{rf['retained']}","poststratum_cells\t432","aggregate_master_width_histogram\t"+",".join(map(str,aggregate)),f"probe_orbit_rejections\t{df['probe_orbit_rejections']}",f"pooled_ge64_observed\t{rf['pooled']}",f"records\t{len(sel)}"]
  for rd,pb,fr in vector_bases():
   for g in VECTOR_GROUPS:
    cell=f"r{rd}:p{pb:02d}:f{fr}:g{g}"; lines.append(f"poststratum\t{cell}\t{N[cell]}\t{q[cell]}\t{N[cell]}\t{total}\t"+",".join(map(str,widths[cell])))
  lines.append("columns\tallocation_id\tsource_match_index\tsource_state_index\tsource_match_id\tunit\tround\tply_stratum\tfrontier_present\tallocation_slot\tpost_stratum\tmaster_width\tcensus_count\tallocation_quota\tweight_numerator\tweight_denominator\torbit_sha256\tstate_sha256\tallocation_priority_sha256\tmask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\tdiscovery_sha256")
  for i,r in enumerate(sel):
   cell=r["cell"]; unit=f"{r['source_match_id']}:s{int(r['source_state_index']):03d}"
   lines.append("\t".join((str(i),r["source_match_index"],r["source_state_index"],r["source_match_id"],unit,r["round"],r["ply_bin"],r["frontier_present"],r["allocation_slot"],cell,r["master_width"],str(N[cell]),str(q[cell]),str(N[cell]),str(q[cell]*total),r["orbit_sha256"],r["state_sha256"],r["priority_sha256"],r["mask_001_sha256"],r["mask_002_sha256"],r["master_sha256"],r["state_hex"],dhx)))
 return ("\n".join(lines)+"\n").encode("ascii")

def main()->int:
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--discovery",required=True,type=Path)
 p.add_argument("--discovery-sha256",required=True)
 p.add_argument("--reservoir",required=True,type=Path)
 p.add_argument("--reservoir-sha256",required=True)
 p.add_argument("--out",type=Path)
 p.add_argument("--validate-contract-only","--contract-smoke",
                dest="validate_contract_only",action="store_true")
 p.add_argument("--split",choices=tuple(SEEDS))
 p.add_argument("--smoke-seed",type=int)
 p.add_argument("--matches",type=int)
 p.add_argument("--reservoir-per-cell",type=int)
 a=p.parse_args()
 try:
  if a.validate_contract_only:
   if a.out is not None or None in (
       a.split,a.smoke_seed,a.matches,a.reservoir_per_cell):
    raise AllocationError(
     "contract smoke requires split/seed/matches/cap and forbids --out")
   digest=validate_contract_smoke(
    a.discovery,a.discovery_sha256,a.reservoir,a.reservoir_sha256,
    split=a.split,seed=a.smoke_seed,matches=a.matches,
    reservoir_per_cell=a.reservoir_per_cell)
   print(f"contract_smoke_sha256={digest}")
   return 0
  if a.out is None or any(value is not None for value in (
      a.split,a.smoke_seed,a.matches,a.reservoir_per_cell)):
   raise AllocationError("campaign allocation requires --out only")
  data=build_manifest(
   a.discovery,a.discovery_sha256,a.reservoir,a.reservoir_sha256)
  atomic(a.out,data)
 except (OSError,ValueError,AllocationError) as e: p.error(str(e))
 print(f"allocation_sha256={hashlib.sha256(data).hexdigest()}");return 0

if __name__=="__main__": raise SystemExit(main())
