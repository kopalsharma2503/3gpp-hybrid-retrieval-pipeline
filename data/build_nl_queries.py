

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "real_corpus.jsonl")
OUT = os.path.join(HERE, "nl_queries.jsonl")

# (natural-language question, spec, target_clause)
NL = [
    # ---- TS 38.331 RRC ----
    ("What connection states can a 5G handset be in, and what makes it move between them?", "38.331", "4.2.1"),
    ("How is over-the-air security switched on for a radio connection?", "38.331", "5.3.1.2"),
    ("How does a device decide that its radio link has been lost?", "38.331", "5.3.10.3"),
    ("What does a phone do when lower-layer trouble clears up before a failure is declared?", "38.331", "5.3.10.2"),
    ("What does a handset do when it receives the message that brings its suspended connection back?", "38.331", "5.3.13.4"),
    ("What does a device clean up when it drops back to the idle state?", "38.331", "5.3.11"),
    ("How does a phone read the broadcast configuration of a cell it camps on?", "38.331", "5.2.2.3"),
    ("What happens if a device is unable to bring its suspended connection back?", "38.331", "5.3.13.5"),
    ("Which control-plane bearers carry signalling between the device and the base station?", "38.331", "4.2.2"),
    ("What request does a device send to wake a suspended connection back up?", "38.331", "5.3.13.3"),
    ("What does a device do right after it receives a fresh broadcast information block?", "38.331", "5.2.2.4"),

    # ---- TS 38.300 NR overall ----
    ("How does the base station decide which users get airtime on the downlink?", "38.300", "10.2"),
    ("How are permissions to transmit on the uplink handed out to devices?", "38.300", "10.3"),
    ("How can a cell cut power use by going quiet during light traffic?", "38.300", "15.4.2.3"),
    ("How does the network spot a handover that ended in a dropped connection?", "38.300", "15.5.2.2"),
    ("How are extra serving cells switched on and off for a connected device?", "38.300", "10.6"),
    ("How can transmissions on one carrier be scheduled from a different carrier?", "38.300", "10.8"),
    ("What inputs does the scheduler use to decide resource allocation?", "38.300", "10.4"),

    # ---- TS 38.321 MAC ----
    ("What makes a device begin the procedure to get onto the uplink from scratch?", "38.321", "5.1.1"),
    ("How does a device pick which preamble and resources to use to access a cell?", "38.321", "5.1.2"),
    ("How does a device send its access preamble to the network?", "38.321", "5.1.3"),
    ("What reply does a device wait for after sending its access preamble?", "38.321", "5.1.4"),
    ("How is a clash sorted out when two devices happen to choose the same access preamble?", "38.321", "5.1.5"),
    ("What does a device do when the beam that was serving it disappears?", "38.321", "5.17"),
    ("What does the MAC layer wipe when it is told to reset?", "38.321", "5.12"),
    ("How is packet copying turned on to make a bearer more reliable?", "38.321", "5.10"),
    ("How does the MAC layer cope with the gaps reserved for taking measurements?", "38.321", "5.14"),

    # ---- TS 23.501 5GC architecture ----
    ("Which functional building blocks make up the core of a 5G network?", "23.501", "4.2.2"),
    ("How is the user-plane traffic of a data session kept protected?", "23.501", "5.10.3"),
    ("How do 5G core functions offer their capabilities to one another?", "23.501", "4.2.6"),
    ("What are the named links drawn between the different core network functions?", "23.501", "4.2.7"),
    ("How does the core support a device attached to two radio nodes at the same time?", "23.501", "5.11.1"),
    ("How does the 5G system make voice calls over IMS possible?", "23.501", "4.4.3"),
    ("How is the data volume carried on a secondary radio technology reported for billing?", "23.501", "5.12.2"),
    ("How is subscriber and network state kept and shared across the core?", "23.501", "4.2.5"),
    ("What does the 5G architecture look like when the subscriber is on their home network?", "23.501", "4.2.3"),

    # ---- TS 24.501 NAS ----
    ("How is signalling between the device and the core encrypted and decrypted?", "24.501", "4.4.3.4"),
    ("How does a device check that a signalling message it received was not altered?", "24.501", "4.4.4.2"),
    ("How is a protected signalling channel set up between the device and the AMF?", "24.501", "4.4.2.5"),
    ("At what point does the network begin encrypting its signalling messages?", "24.501", "4.4.5"),
    ("How does a device choose between the packet and circuit domains to place a call?", "24.501", "4.3.2"),
    ("What is done when the signalling message counter rolls over to its limit?", "24.501", "4.4.3.5"),
    ("How does the network hand applications keys that come from the SIM credentials?", "24.501", "4.21"),
    ("How is repeatedly re-sending a device's radio capabilities avoided?", "24.501", "4.16"),

    # ---- TS 33.501 Security ----
    ("How does an operator load the protection rules onto the edge security proxy?", "33.501", "13.2.3.5"),
    ("How is signalling kept safe as it crosses between two operators' edge proxies?", "33.501", "13.1.2"),
    ("How do the exposure function and an outside application prove who they are to each other?", "33.501", "12.2"),
    ("How are keys created for an emergency call placed by a device that is not authenticated?", "33.501", "10.2.2.3"),
    ("How is the payload of inter-operator signalling encrypted at the border proxy?", "33.501", "13.2.4.4"),
    ("How does the border security proxy prove its identity to internal network functions?", "33.501", "13.3.3"),
    ("How is a device's secondary data-network authorization taken away?", "33.501", "11.1.4"),
]


def build():
    corpus = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    by_clause = {}
    for d in corpus:
        by_clause.setdefault((d["spec"], d["clause"]), []).append(d["id"])

    written = 0
    missing = []
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (q, spec, clause) in enumerate(NL):
            gold = by_clause.get((spec, clause))
            if not gold:
                missing.append((spec, clause, q))
                continue
            f.write(json.dumps({
                "qid": f"n{i:03d}", "query": q,
                "relevant": gold, "spec": spec, "clause": clause,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} natural-language queries -> {OUT}")
    if missing:
        print(f"WARNING: {len(missing)} target clauses not found in corpus:")
        for spec, clause, q in missing:
            print(f"  TS {spec} clause {clause}  ::  {q}")


if __name__ == "__main__":
    build()
