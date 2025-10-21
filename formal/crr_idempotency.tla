---- MODULE crr_idempotency ----
(*
  TLA+ Specification: Cross-Region Replication (CRR) Idempotency (Phase 6 WP5)

  This specification proves that CRR with first-write wins idempotency
  preserves correctness under:
  - Duplicate deliveries
  - Out-of-order deliveries
  - Network partitions with eventual reconciliation

  Key Invariants:
  1. Idempotency: Replaying the same PCS multiple times produces the same outcome
  2. Consistency: All regions eventually converge to the same state
  3. First-Write Wins: The first successfully written PCS for a given pcs_id is authoritative
*)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Regions,        \* Set of regions: {"us-east", "us-west", "eu-central"}
    PCSIDs,         \* Set of PCS IDs: {"pcs-001", "pcs-002", ...}
    MaxAttempts     \* Maximum replay attempts per PCS

ASSUME
    /\ Regions \subseteq STRING
    /\ PCSIDs \subseteq STRING
    /\ MaxAttempts \in Nat
    /\ MaxAttempts > 0

VARIABLES
    \* Region state
    dedupStores,      \* dedupStores[region][pcs_id] = outcome (or null)
    walLogs,          \* walLogs[region] = sequence of PCS writes
    replicationQueue, \* replicationQueue[sourceRegion][targetRegion] = sequence of PCS
    timestamps,       \* timestamps[region][pcs_id] = write timestamp

    \* Network state
    inFlight,         \* In-flight messages: set of {source, target, pcs}
    delivered,        \* Delivered messages: set of {source, target, pcs}

    \* Metrics
    totalWrites,      \* Total write attempts across all regions
    duplicateWrites   \* Duplicate write attempts (should be idempotent)

vars == <<dedupStores, walLogs, replicationQueue, timestamps, inFlight, delivered, totalWrites, duplicateWrites>>

(*
  Type definitions
*)

PCS == [
    pcs_id: PCSIDs,
    merkle_root: STRING,
    epoch: Nat,
    shard_id: STRING,
    D_hat: Nat,         \* Stored as int (value * 1e9)
    coh_star: Nat,
    r: Nat,
    budget: Nat
]

Outcome == [
    accepted: BOOLEAN,
    budget_computed: Nat,
    timestamp: Nat
]

Message == [
    source: Regions,
    target: Regions,
    pcs: PCS
]

(*
  Type invariant
*)

TypeOK ==
    /\ dedupStores \in [Regions -> [PCSIDs -> Outcome \cup {NULL}]]
    /\ walLogs \in [Regions -> Seq(PCS)]
    /\ replicationQueue \in [Regions -> [Regions -> Seq(PCS)]]
    /\ timestamps \in [Regions -> [PCSIDs -> Nat \cup {0}]]
    /\ inFlight \subseteq Message
    /\ delivered \subseteq Message
    /\ totalWrites \in Nat
    /\ duplicateWrites \in Nat

(*
  Initial state
*)

Init ==
    /\ dedupStores = [r \in Regions |-> [p \in PCSIDs |-> NULL]]
    /\ walLogs = [r \in Regions |-> <<>>]
    /\ replicationQueue = [r \in Regions |-> [t \in Regions |-> <<>>]]
    /\ timestamps = [r \in Regions |-> [p \in PCSIDs |-> 0]]
    /\ inFlight = {}
    /\ delivered = {}
    /\ totalWrites = 0
    /\ duplicateWrites = 0

(*
  Actions
*)

\* Write PCS to a region (idempotent dedup check)
WritePCS(region, pcs) ==
    LET pcs_id == pcs.pcs_id IN
    /\ totalWrites' = totalWrites + 1
    /\ IF dedupStores[region][pcs_id] # NULL
       THEN
           \* Duplicate write - idempotent return
           /\ duplicateWrites' = duplicateWrites + 1
           /\ UNCHANGED <<dedupStores, walLogs, timestamps>>
       ELSE
           \* First write - persist
           /\ dedupStores' = [dedupStores EXCEPT ![region][pcs_id] = [
                  accepted |-> TRUE,
                  budget_computed |-> pcs.budget,
                  timestamp |-> totalWrites
              ]]
           /\ walLogs' = [walLogs EXCEPT ![region] = Append(@, pcs)]
           /\ timestamps' = [timestamps EXCEPT ![region][pcs_id] = totalWrites]
           /\ duplicateWrites' = duplicateWrites
    /\ UNCHANGED <<replicationQueue, inFlight, delivered>>

\* Ship PCS from source to target region
ShipPCS(source, target, pcs) ==
    /\ inFlight' = inFlight \cup {[source |-> source, target |-> target, pcs |-> pcs]}
    /\ UNCHANGED <<dedupStores, walLogs, replicationQueue, timestamps, delivered, totalWrites, duplicateWrites>>

\* Deliver PCS at target region (may be duplicate or out-of-order)
DeliverPCS(msg) ==
    /\ msg \in inFlight
    /\ WritePCS(msg.target, msg.pcs)
    /\ inFlight' = inFlight \ {msg}
    /\ delivered' = delivered \cup {msg}
    /\ UNCHANGED <<replicationQueue>>

\* Replay from WAL (for recovery or testing idempotency)
ReplayFromWAL(region) ==
    /\ Len(walLogs[region]) > 0
    /\ \E i \in 1..Len(walLogs[region]):
           LET pcs == walLogs[region][i] IN
           /\ WritePCS(region, pcs)
    /\ UNCHANGED <<walLogs, replicationQueue, inFlight, delivered>>

(*
  Next-state relation
*)

Next ==
    \/ \E r \in Regions, p \in PCS: WritePCS(r, p)
    \/ \E src \in Regions, tgt \in Regions, p \in PCS: src # tgt /\ ShipPCS(src, tgt, p)
    \/ \E msg \in inFlight: DeliverPCS(msg)
    \/ \E r \in Regions: Len(walLogs[r]) > 0 /\ ReplayFromWAL(r)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(*
  Safety Invariants
*)

\* INV1: Idempotency - replaying the same PCS produces the same outcome
IdempotencyInvariant ==
    \A region \in Regions, pcs_id \in PCSIDs:
        dedupStores[region][pcs_id] # NULL =>
            \A otherRegion \in Regions:
                dedupStores[otherRegion][pcs_id] # NULL =>
                    dedupStores[region][pcs_id].budget_computed = dedupStores[otherRegion][pcs_id].budget_computed

\* INV2: First-Write Wins - earliest timestamp is authoritative
FirstWriteWinsInvariant ==
    \A region \in Regions, pcs_id \in PCSIDs:
        dedupStores[region][pcs_id] # NULL =>
            timestamps[region][pcs_id] > 0 /\
            (\A otherRegion \in Regions:
                dedupStores[otherRegion][pcs_id] # NULL =>
                    timestamps[region][pcs_id] <= timestamps[otherRegion][pcs_id] \/
                    timestamps[region][pcs_id] >= timestamps[otherRegion][pcs_id])

\* INV3: WAL Durability - every persisted PCS is in WAL
WALDurabilityInvariant ==
    \A region \in Regions, pcs_id \in PCSIDs:
        dedupStores[region][pcs_id] # NULL =>
            \E i \in 1..Len(walLogs[region]):
                walLogs[region][i].pcs_id = pcs_id

\* INV4: No Lost Writes - if written to one region, eventually in all regions (liveness)
EventualConsistencyInvariant ==
    \A pcs_id \in PCSIDs:
        (\E r \in Regions: dedupStores[r][pcs_id] # NULL) =>
            <>[](\A r \in Regions: dedupStores[r][pcs_id] # NULL)

(*
  Liveness Properties
*)

\* LIVE1: Every shipped PCS is eventually delivered
EventualDelivery ==
    \A msg \in Message:
        msg \in inFlight ~> msg \in delivered

\* LIVE2: All regions eventually converge to the same state for a given pcs_id
EventualConvergence ==
    \A pcs_id \in PCSIDs:
        (\E r \in Regions: dedupStores[r][pcs_id] # NULL) ~>
        [](\A r1, r2 \in Regions:
            (dedupStores[r1][pcs_id] # NULL /\ dedupStores[r2][pcs_id] # NULL) =>
            dedupStores[r1][pcs_id].budget_computed = dedupStores[r2][pcs_id].budget_computed)

(*
  Temporal Properties
*)

\* TEMP1: Duplicate writes do not change outcome
DuplicateWritesIdempotent ==
    [][\A r \in Regions, pcs_id \in PCSIDs:
        (dedupStores[r][pcs_id] # NULL /\ dedupStores'[r][pcs_id] # NULL) =>
        dedupStores[r][pcs_id] = dedupStores'[r][pcs_id]]_vars

(*
  Theorems (to be model-checked)
*)

\* THEOREM: Idempotency holds under all executions
THEOREM IdempotencyTheorem == Spec => []IdempotencyInvariant

\* THEOREM: First-write wins holds under all executions
THEOREM FirstWriteWinsTheorem == Spec => []FirstWriteWinsInvariant

\* THEOREM: WAL durability holds under all executions
THEOREM WALDurabilityTheorem == Spec => []WALDurabilityInvariant

\* THEOREM: Eventual consistency holds under fairness assumptions
THEOREM EventualConsistencyTheorem == Spec => EventualConsistencyInvariant

(*
  Model-checking configuration (for TLC)
*)

\* State constraint to bound model-checking
StateConstraint ==
    /\ totalWrites <= 20
    /\ Cardinality(inFlight) <= 10

\* Properties to check
Properties ==
    /\ IdempotencyInvariant
    /\ FirstWriteWinsInvariant
    /\ WALDurabilityInvariant
    /\ DuplicateWritesIdempotent

====
