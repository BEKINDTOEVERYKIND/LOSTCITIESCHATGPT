# Belief-history v1 result

The frozen history-aware hand model passed the complete untouched TEST accuracy
gate. It is retained as the best belief-accuracy artifact, not as a playing
actor. No playing-strength claim or actor promotion follows from this result.

The panel comprised 65,536 source matches, 9,119,990 scored states, and
274,351,522 uncertain-card labels. Against the maintained incumbent belief
head, history improved all-state joint NLL from 13.2417657213 to 12.7827832218,
post-opponent-action joint NLL from 13.0831905964 to 12.6140953805, and
all-state Brier from 0.1618884254 to 0.1559471996. Every frozen one-sided 99%
simultaneous familywise lower bound was strictly positive. History also passed
the same complete bundle directly against the matched-budget head-only control.

GitHub run 33253283912 failed only while packaging the already-complete result:
`mkdir python-runtime complete/raw` did not create the missing `complete`
parent. All sixteen TEST shards, the training freeze, the source-free runtime,
artifact ZIP digests, and inner manifests were independently verified. Replaying
the sealed reducer produced canonical verdict SHA-256
`21eb3961b43c9c5bd98dcddc909f0c0001d63c07c04d040740e5cbc3c03bb929`.
TRAIN and TEST are not rerun; all campaign roots remain retired.

The selected standalone model is stored losslessly as
`data/models/belief_history_v1_accuracy.lcbhm.gz`, archive SHA-256
`166dacecbb543156d69fcfd7ad1c36d611a693283ceef2316fc9931f5fbd7923`.
Its decompressed model SHA-256 is
`c07ccb5f58d38a5a086d67e17904461c266f1acb8060511bdd4ddc9b5bbd7581`.
Any integration into rollout-world construction is a separate playing candidate
and must pass the unchanged reciprocal safety and final match gates.
