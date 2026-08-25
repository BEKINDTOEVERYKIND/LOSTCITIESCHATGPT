/* White-box contract tests for the new-file-only offline dataset tool. */
#define main policy_cost_dataset_program_main
#include "../tools/policy_cost_dataset.c"
#undef main

#include <stdio.h>

static int failures;
#define CHECK(x, msg) do { if (!(x)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } } while (0)

static Agent new_controller(const Net *net)
{
    Agent a;agent_default(&a,AG_ROLLOUT,net);a.continuation_net=net;
    a.dets=800;a.confirm_dets=800;a.root_width=5;a.action_core_count=3;
    a.min_cand=1;a.ply_lo=0;a.ply_hi=0;a.cand_floor=0.01f;
    a.override_k=3.5f;a.override_min=0.0f;a.playout_sample=4;
    a.symmetries=20;a.playout_symmetries=20;a.discard_guard=1;
    a.playout_prune=1;a.policy_prefix_mode=0;a.no_belief=1;
    a.belief_alpha=1.0f;a.exact_terminal=1;a.bounded_late_min=1.0f;
    return a;
}

static PolicyCostTable table_for(const Agent *a)
{
    static const uint32_t anchor[POLICY_COST_ANCHORS]={0,4,8,12,16,24,32,40,48,64};
    PolicyCostTable t;memset(&t,0,sizeof t);t.version=POLICY_COST_VERSION;
    t.source_seed=UINT64_C(202611140101);t.payload_fingerprint=1;t.epsilon=POLICY_COST_EPSILON;
    t.primary_z=POLICY_COST_PRIMARY_Z;t.fresh_z=POLICY_COST_FRESH_Z;
    t.controller=(PolicyCostController){
        .root_net_fingerprint=match_value_net_fingerprint(a->net),
        .continuation_net_fingerprint=match_value_net_fingerprint(a->continuation_net),
        .controller_abi=POLICY_COST_CONTROLLER_ABI,
        .build_profile=match_value_build_profile(),.objective=0,
        .root_symmetries=20,.playout_symmetries=20,.playout_sample=4,
        .playout_prune=1,.exact_terminal=1,.no_belief=1,.dets=800,
        .confirm_dets=800,.root_width=5,.action_core_count=3,.min_cand=1,
        .ply_lo=0,.ply_hi=0,.discard_guard=1,.root_prune=0,
        .cand_floor=0.01f,.override_k=3.5f,.override_min=0.0f,
    };
    for(int i=0;i<POLICY_COST_ANCHORS;i++){t.ply_anchor[i]=anchor[i];
        t.lambda_action[i]=1.0;t.lambda_draw[i]=0.5;}
    return t;
}

static MatchValueTable *objective3_match_fixture(const Net *net)
{
    MatchValueTable *table=(MatchValueTable *)calloc(1,sizeof *table);
    if(!table)return NULL;
    table->version=MATCH_VALUE_VERSION;
    table->samples_per_policy_lead=400;
    table->role_cycle_size=400;
    table->role_balance_complete=1;
    table->isotonic_projected=1;
    table->source_seed=UINT64_C(7331001);
    table->payload_fingerprint=UINT64_C(0x123456789abcdef0);
    table->controller=(MatchValueController){
        .net_fingerprint=match_value_net_fingerprint(net),
        .controller_abi=MATCH_VALUE_CONTROLLER_ABI,
        .build_profile=match_value_build_profile(),
        .objective=0,.playout_symmetries=20,.playout_sample=4,
        .playout_prune=1,.exact_terminal=1,.max_plies=LC_MAX_PLIES,
    };
    return table;
}

static void test_bound_reservoir_index(void)
{
    enum { ROWS=4096 };
    BoundReservoir reservoir={0};reservoir.is_vector=1;
    reservoir.n=reservoir.allocated=ROWS;
    reservoir.row=(BoundReservoirRow *)calloc(ROWS,sizeof *reservoir.row);
    CHECK(reservoir.row,"allocate indexed-reservoir fixture");
    if(!reservoir.row)return;
    for(int i=0;i<ROWS;i++){
        /* Reverse source order proves the one-time sort is doing work. */
        Allocation *a=&reservoir.row[i].value;
        a->source_match_index=(uint64_t)(ROWS-i);
        a->source_state_index=i%97;a->round=i%3;a->ply_bin=i%24;
        a->frontier=i%2;a->allocation_slot=0;a->master_width=1;
        snprintf(a->cell,sizeof a->cell,"r%d:p%02d:f%d:j0",
                 a->round,a->ply_bin,a->frontier);
        snprintf(a->source_match_id,sizeof a->source_match_id,
                 "SELECT-%012llu",(unsigned long long)a->source_match_index);
        a->state_n=1;a->state_bytes[0]=(unsigned char)i;
        memset(a->priority,(unsigned char)(i+1),sizeof a->priority);
        memset(a->state_hash,(unsigned char)(i+3),sizeof a->state_hash);
    }
    CHECK(bound_reservoir_index(&reservoir),"cannot build reservoir index");
    CHECK(reservoir.indexed,"reservoir did not record indexed state");
    for(int i=0;i<ROWS;i+=127){
        Allocation exact=reservoir.row[i].value;
        int comparisons=0;
        CHECK(allocation_index_position(&exact,&reservoir,&comparisons)==i,
              "indexed lookup missed exact row");
        CHECK(comparisons<=14,
              "4096-row lookup exceeded logarithmic comparison bound");
        CHECK(allocation_in_reservoir(&exact,&reservoir),
              "indexed lookup rejected exact row content");
        exact.state_hash[0]^=1;
        CHECK(!allocation_in_reservoir(&exact,&reservoir),
              "indexed lookup ignored content drift");
    }
    Allocation absent=reservoir.row[0].value;
    absent.source_match_index=UINT64_MAX;
    CHECK(!allocation_in_reservoir(&absent,&reservoir),
          "indexed lookup accepted absent allocation");

    BoundReservoir duplicate={0};duplicate.is_vector=1;duplicate.n=2;
    duplicate.allocated=2;duplicate.row=(BoundReservoirRow *)calloc(2,sizeof *duplicate.row);
    CHECK(duplicate.row,"allocate duplicate-index fixture");
    if(duplicate.row){
        duplicate.row[0].value=reservoir.row[0].value;
        duplicate.row[1].value=reservoir.row[0].value;
        CHECK(!bound_reservoir_index(&duplicate),
              "duplicate reservoir allocation identity was indexed");
    }
    bound_reservoir_free(&duplicate);bound_reservoir_free(&reservoir);
}

static void test_reservoir_manifest_counts(void)
{
    AllocationManifest manifest={0};BoundReservoir reservoir={0};
    manifest.is_vector=1;manifest.total_census=100;manifest.retained_units=40;
    reservoir.is_vector=1;reservoir.eligible=100;reservoir.retained=40;
    CHECK(reservoir_counts_match_manifest(&reservoir,&manifest),
          "valid vector reservoir/manifest counts rejected");
    reservoir.eligible=99;
    CHECK(!reservoir_counts_match_manifest(&reservoir,&manifest),
          "vector eligible census drift accepted");
    reservoir.eligible=100;reservoir.retained=39;
    CHECK(!reservoir_counts_match_manifest(&reservoir,&manifest),
          "vector retained census drift accepted");
    memset(&manifest,0,sizeof manifest);memset(&reservoir,0,sizeof reservoir);
    manifest.is_vector=0;manifest.eligible_units=200;manifest.retained_units=80;
    reservoir.is_vector=0;reservoir.eligible=200;reservoir.retained=80;
    CHECK(reservoir_counts_match_manifest(&reservoir,&manifest),
          "valid TRAIN reservoir/manifest counts rejected");
    reservoir.eligible++;
    CHECK(!reservoir_counts_match_manifest(&reservoir,&manifest),
          "TRAIN eligible census drift accepted");
}

static void test_source_match_ranges(void)
{
    CHECK(source_match_in_range(SPLIT_TRAIN,UINT64_C(65535))&&
          !source_match_in_range(SPLIT_TRAIN,UINT64_C(65536)),
          "TRAIN reader source range is not the frozen 65,536 matches");
    CHECK(source_match_in_range(SPLIT_SELECT,UINT64_C(32767))&&
          !source_match_in_range(SPLIT_SELECT,UINT64_C(32768))&&
          source_match_in_range(SPLIT_TEST,UINT64_C(32767))&&
          !source_match_in_range(SPLIT_TEST,UINT64_C(32768)),
          "holdout reader source range is not the frozen 32,768 matches");
}

static void test_interleaved_allocation_schedule(void)
{
    const int train_records=3*PC_PLY_BINS*PC_RATIO_BINS*PC_PAIR_TYPES*
        PC_TRAIN_QUOTA;
    for(int start=0;start<train_records;start+=64){
        int round_seen[3]={0},ply_seen[PC_PLY_BINS]={0};
        int ratio_seen[PC_RATIO_BINS]={0},type_seen[PC_PAIR_TYPES]={0};
        int count=0;
        for(int row=start;row<start+64;row++){
            int rd,pb,ratio,type;
            train_scheduled_cell(row,&rd,&pb,&ratio,&type);
            round_seen[rd]=ply_seen[pb]=ratio_seen[ratio]=type_seen[type]=1;
            count++;
        }
        int rounds=0,plies=0,ratios=0,types=0;
        for(int i=0;i<3;i++)rounds+=round_seen[i];
        for(int i=0;i<PC_PLY_BINS;i++)plies+=ply_seen[i];
        for(int i=0;i<PC_RATIO_BINS;i++)ratios+=ratio_seen[i];
        for(int i=0;i<PC_PAIR_TYPES;i++)types+=type_seen[i];
        CHECK(count==64&&rounds==3&&plies==24&&ratios==6&&types==2,
              "TRAIN 64-row schedule lost fixed-factor coverage");
    }
    for(int start=0;start<3*PC_PLY_BINS*2*PC_VECTOR_QUOTA;start+=48){
        int round_seen[3]={0},frontier_seen[2]={0},band_seen[3]={0};
        int ply_seen[PC_PLY_BINS]={0};
        for(int row=start;row<start+48;row++){
            int rd,pb,frontier;
            vector_scheduled_base(row,&rd,&pb,&frontier);
            round_seen[rd]=frontier_seen[frontier]=band_seen[pb/8]=1;
            ply_seen[pb]=1;
        }
        int rounds=0,frontiers=0,bands=0,plies=0;
        for(int i=0;i<3;i++){rounds+=round_seen[i];bands+=band_seen[i];}
        for(int i=0;i<2;i++)frontiers+=frontier_seen[i];
        for(int i=0;i<PC_PLY_BINS;i++)plies+=ply_seen[i];
        CHECK(rounds==3&&frontiers==2&&bands==3&&plies==8,
              "vector 48-row schedule lost round/frontier/ply-band balance");
    }
    Allocation first={0},second={0};
    first.priority[31]=1;second.priority[31]=2;
    CHECK(allocation_tuple_before(&first,&second)&&
          !allocation_tuple_before(&second,&first)&&
          !allocation_tuple_before(&first,&first),
          "native within-cell allocation tuple is not strict");
}

static void test_master_panel_projection(void)
{
    RuntimeMask master_mask={.n=5},subset_mask={.n=3};
    const int master_index[5]={10,20,30,40,50};
    const int subset_index[3]={10,30,50};
    memcpy(master_mask.index,master_index,sizeof master_index);
    memcpy(subset_mask.index,subset_index,sizeof subset_index);
    RolloutAuditPanel master={0},subset={0};
    master.n=5;master.requested_worlds=800;master.worlds=137;
    master.baseline=0;master.selected=3;master.exact_hidden_support=1;
    master.hidden_support=137;master.objective=0;
    master.panel_role=ROLLOUT_AUDIT_PANEL_PRIMARY;
    master.hidden_world_fingerprint=UINT64_C(0x123456789abcdef0);
    master.exact_terminal_leaves=91;master.cycle_breaks=17;
    for(int i=0;i<5;i++){
        master.q[i]=(double)(i+1);master.se[i]=(double)i/10.0;
        master.delta[i]=(double)(10+i);master.delta_se[i]=(double)(20+i);
        for(int j=0;j<5;j++){
            master.pair_delta[i][j]=(double)(100+10*i+j);
            master.pair_delta_se[i][j]=(double)(200+10*i+j);
        }
    }
    CHECK(derive_audit_subset(&master,&master_mask,&subset_mask,&subset),
          "cannot derive no-refill submatrix from master panel");
    CHECK(subset.n==3&&subset.baseline==0&&subset.selected==2&&
          subset.requested_worlds==800&&subset.worlds==137&&
          subset.hidden_world_fingerprint==master.hidden_world_fingerprint&&
          subset.exact_terminal_leaves==master.exact_terminal_leaves&&
          subset.q[0]==master.q[0]&&subset.q[1]==master.q[2]&&
          subset.q[2]==master.q[4]&&
          subset.pair_delta[1][2]==master.pair_delta[2][4]&&
          subset.pair_delta_se[2][1]==master.pair_delta_se[4][2],
          "derived submatrix changed master action/pair evidence");
    RuntimeMask reordered=subset_mask;
    reordered.index[1]=50;reordered.index[2]=30;
    CHECK(!derive_audit_subset(&master,&master_mask,&reordered,&subset),
          "reordered/refilled mask was accepted as a master submatrix");

    RolloutAuditPanel primary=master,fresh=master;
    primary.panel_role=ROLLOUT_AUDIT_PANEL_PRIMARY;
    fresh.panel_role=ROLLOUT_AUDIT_PANEL_FRESH;
    CHECK(independent_hidden_panel_or_complete_census(&primary,&fresh),
          "equal complete finite censuses were rejected as sample reuse");
    primary.worlds=primary.hidden_support=800;
    fresh.worlds=fresh.hidden_support=800;
    CHECK(!independent_hidden_panel_or_complete_census(&primary,&fresh),
          "equal random 800-world panels were accepted");
    primary.worlds=primary.hidden_support=137;
    fresh.worlds=fresh.hidden_support=137;fresh.exact_hidden_support=0;
    CHECK(!independent_hidden_panel_or_complete_census(&primary,&fresh),
          "equal partial finite-support panels were accepted");
    fresh.exact_hidden_support=1;fresh.hidden_world_fingerprint++;
    CHECK(independent_hidden_panel_or_complete_census(&primary,&fresh),
          "distinct hidden-world panels were rejected");
}

static void test_retained_origin_mutations(const Net *net)
{
    State complete,view;Rng rng;rng_seed(&rng,UINT64_C(424242));
    lc_deal(&complete,&rng);agent_information_view(&complete,complete.turn,&view);
    Allocation a;memset(&a,0,sizeof a);a.source_match_index=32767;
    a.source_state_index=899;a.round=view.round;a.ply_bin=ply_bin(view.nply);
    CHECK(encode_view(&view,a.state_bytes,&a.state_n)&&a.state_n==174,
          "cannot encode retained-origin fixture");
    sha_bytes(a.state_bytes,a.state_n,a.state_hash);
    CHECK(orbit_digest(&view,a.orbit),"cannot hash retained-origin fixture");
    Move mv[MAX_MOVES];float prob[MAX_MOVES];RuntimeMask mask[PC_MASKS];
    int n=0,baseline=0,uindex[PC_UNION_MAX],nunion=0;
    CHECK(policy_snapshot(net,&view,mv,prob,&n,&baseline,mask,uindex,&nunion),
          "cannot build retained-origin policy fixture");
    a.master_width=mask[0].n;a.frontier=frontier_present(mv,prob,n,mask);
    a.allocation_slot=0;memcpy(a.mask_hash[0],mask[0].hash,32);
    memcpy(a.mask_hash[1],mask[1].hash,32);memcpy(a.union_hash,mask[0].hash,32);
    Exclusions none={0};
    CHECK(verify_retained_origin(&a,net,&none),
          "valid retained native origin rejected");
    Allocation bad=a;bad.state_n=173;
    CHECK(!verify_retained_origin(&bad,net,&none),"short state bytes accepted");
    bad=a;bad.source_match_index=32768;
    CHECK(!verify_retained_origin(&bad,net,&none),"out-of-range source accepted");
    bad=a;bad.source_state_index=900;
    CHECK(!verify_retained_origin(&bad,net,&none),"out-of-range state accepted");
    bad=a;bad.orbit[0]^=1;
    CHECK(!verify_retained_origin(&bad,net,&none),"false orbit accepted");
    bad=a;bad.mask_hash[0][0]^=1;
    CHECK(!verify_retained_origin(&bad,net,&none),"false policy mask accepted");
    bad=a;bad.ply_bin=(bad.ply_bin+1)%PC_PLY_BINS;
    CHECK(!verify_retained_origin(&bad,net,&none),"false ply cell accepted");
    Exclusions blocked={0};blocked.n=1;memcpy(blocked.hash[0],a.orbit,32);
    CHECK(!verify_retained_origin(&a,net,&blocked),"exact17 orbit accepted");
}

int main(void)
{
    test_bound_reservoir_index();
    test_reservoir_manifest_counts();
    test_source_match_ranges();
    test_interleaved_allocation_schedule();
    test_master_panel_projection();
    Net *net=(Net *)malloc(sizeof *net);
    CHECK(net&&net_load(net,"data/champion.bin")==0,"load champion fixture");
    if(!net)return 1;
    Agent a=new_controller(net);PolicyCostTable t=table_for(&a);
    test_retained_origin_mutations(net);
    CHECK(frozen_base_actor(&a,net),"valid neutral rollout5 controller rejected");
    CHECK(table_matches_counterfactual_actor(&t,&a),"valid table/controller rejected");
    a.cand_floor=0.02f;a.ply_lo=14;
    CHECK(table_matches_counterfactual_actor(&t,&a),"preregistered floor/onset counterfactual rejected");
    PolicyCostTable bad=t;bad.controller.continuation_net_fingerprint^=1;
    CHECK(!table_matches_counterfactual_actor(&bad,&a),"continuation mismatch accepted");
    bad=t;bad.controller.objective=3;bad.controller.match_value_fingerprint=1;
    CHECK(!table_matches_counterfactual_actor(&bad,&a),"objective/table mismatch accepted");
    a=new_controller(net);a.override_min=2.0f;
    CHECK(!frozen_base_actor(&a,net),"legacy practical hurdle entered rollout5 controller");
    Net *other=(Net *)malloc(sizeof *other);CHECK(other,"allocate altered continuation net");
    if(other){memcpy(other,net,sizeof *other);other->b3+=1.0f;a=new_controller(net);
        a.continuation_net=other;
        CHECK(!frozen_base_actor(&a,net),"base actor accepted altered continuation net");
    }

    Agent maintained=new_controller(net);maintained.action_core_count=0;
    maintained.policy_prefix_mode=3;maintained.cand_floor=0.02f;
    maintained.ply_lo=14;maintained.override_min=2.0f;
    CHECK(frozen_maintained_actor(&maintained,net),"valid maintained legacy actor rejected");
    maintained.action_core_count=3;
    CHECK(!frozen_maintained_actor(&maintained,net),"new shortlist mislabeled maintained");
    MatchValueTable *objective3=objective3_match_fixture(net);
    CHECK(objective3&&match_value_validate(objective3)&&
          match_value_balanced_roles(objective3),
          "cannot construct objective-3 maintained fixture");
    if(objective3){
        maintained=new_controller(net);maintained.action_core_count=0;
        maintained.policy_prefix_mode=3;maintained.cand_floor=0.02f;
        maintained.ply_lo=0;maintained.override_min=2.0f;
        maintained.win_q=3;maintained.match_value=objective3;
        CHECK(frozen_maintained_actor(&maintained,net),
              "valid all-ply objective-3 maintained actor rejected");
        maintained.ply_lo=14;
        CHECK(!frozen_maintained_actor(&maintained,net),
              "objective-3 actor accepted legacy ply-14 onset");
        maintained.ply_lo=0;maintained.win_q=0;
        CHECK(!frozen_maintained_actor(&maintained,net),
              "objective-0 actor accepted all-ply objective-3 table");
        maintained.match_value=NULL;maintained.ply_lo=0;
        CHECK(!frozen_maintained_actor(&maintained,net),
              "objective-0 maintained actor accepted all-ply onset");
    }
    if(other){maintained=new_controller(net);maintained.action_core_count=0;
        maintained.policy_prefix_mode=3;maintained.cand_floor=0.02f;
        maintained.ply_lo=14;maintained.override_min=2.0f;maintained.continuation_net=other;
        CHECK(!frozen_maintained_actor(&maintained,net),"maintained actor accepted altered continuation net");}

    DiscoveryCensus census;
    CHECK(census_init(&census,1,UINT64_C(202611290999)),"census fixture allocation");
    State st;Rng rng;rng_seed(&rng,9);lc_deal(&st,&rng);
    State view;agent_information_view(&st,st.turn,&view);view.deck_left=1;
    RuntimeMask mask[2];memset(mask,0,sizeof mask);mask[0].n=mask[1].n=1;
    int union_index[1]={0};unsigned char orbit[32]={0},state[PC_STATE_MAX]={0},sh[32]={0};
    discovery_account(&census,&view,mask,1);
    discovery_pairs(&census,202611290999,1,0,&view,NULL,NULL,0,mask,
                    union_index,1,orbit,state,1,sh);
    CHECK(census.exact_terminal_preempted[0]==1,
          "one-card exact roots not labeled intrinsic/non-deployable");
    uint64_t attempts=0;
    for(int r=0;r<3;r++)for(int p=0;p<PC_PLY_BINS;p++)for(int g=0;g<6;g++)for(int q=0;q<2;q++)
        attempts+=census.cell[r][p][g][q].attempted;
    CHECK(attempts==0,"one-card exact root entered allocation reservoir");
    census_free(&census);free(objective3);free(other);free(net);
    if(failures)fprintf(stderr,"%d policy-cost dataset failures\n",failures);
    else puts("policy_cost_dataset contract tests: ok");
    return failures!=0;
}
