"""

"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PASSAGES = [
    # ---- TS 38.331 RRC ----
    ("TS 38.331", "4.2.1", "RRC states in NR",
     "NR defines three RRC states. In RRC_IDLE the UE has no RRC connection and "
     "is known at the tracking-area level; in RRC_INACTIVE the UE keeps its "
     "AS context and RAN-based notification area while releasing the air-interface "
     "connection; in RRC_CONNECTED the UE has a signalling and data radio bearer "
     "with the gNB. Transitions between states are controlled by RRC messages "
     "such as RRCRelease with suspendConfig."),
    ("TS 38.331", "5.3.3", "RRC connection setup procedure",
     "The RRC setup procedure establishes SRB1 and moves the UE to RRC_CONNECTED. "
     "The UE sends RRCSetupRequest on CCCH, the gNB responds with RRCSetup carrying "
     "the master cell group configuration, and the UE confirms with RRCSetupComplete "
     "which includes the selected PLMN and the NAS registration message."),
    ("TS 38.331", "5.3.5", "RRC reconfiguration and handover",
     "RRCReconfiguration is the workhorse message used to modify measurement "
     "configuration, add or release radio bearers, configure secondary cells, and "
     "execute handover. A handover is signalled with reconfigurationWithSync, which "
     "provides the target cell physical configuration, a new C-RNTI and the RACH "
     "resources the UE uses to access the target cell."),
    ("TS 38.331", "5.5.4", "Measurement configuration and reporting",
     "The network configures measurement objects, reporting configurations and "
     "measurement identities. Events A1 to A6 trigger reporting: A3 fires when a "
     "neighbour cell becomes an offset better than the serving cell, while A5 fires "
     "when the serving cell drops below one threshold and a neighbour rises above a "
     "second threshold. Reports carry RSRP, RSRQ and SINR measurements."),
    ("TS 38.331", "5.3.7", "RRC connection re-establishment",
     "After a radio link failure, handover failure or integrity check failure the "
     "UE initiates RRC re-establishment. It selects a suitable cell, sends "
     "RRCReestablishmentRequest with a shortMAC-I authentication token, and if the "
     "target cell has the UE context the connection is restored on SRB1 and keys "
     "are refreshed."),

    # ---- TS 38.300 NR overall ----
    ("TS 38.300", "5.1", "NR numerology and frame structure",
     "NR supports multiple subcarrier spacings of 15, 30, 60, 120 and 240 kHz, "
     "each corresponding to a numerology mu. A radio frame is 10 ms and a subframe "
     "is 1 ms; the number of slots per subframe scales with the numerology so that "
     "higher subcarrier spacing yields shorter slots and lower latency, at the cost "
     "of higher overhead and sensitivity to phase noise."),
    ("TS 38.300", "6.10", "Bandwidth parts (BWP)",
     "A bandwidth part is a contiguous set of resource blocks on a carrier. A UE can "
     "be configured with up to four downlink and four uplink bandwidth parts but only "
     "one is active at a time. BWP switching lets the network adapt bandwidth to "
     "traffic and save UE power by operating on a narrow BWP during low activity."),
    ("TS 38.300", "9.2.6", "Beam management in NR",
     "Beam management establishes and maintains a suitable beam pair between the UE "
     "and gNB. It relies on SSB and CSI-RS measurements for beam selection, beam "
     "refinement and beam failure detection. When beam failure is detected the UE "
     "runs a beam failure recovery procedure using a dedicated PRACH resource."),
    ("TS 38.300", "16.1", "Network slicing support in NG-RAN",
     "The NG-RAN supports network slicing by selecting radio resources and an AMF "
     "based on the S-NSSAI signalled by the UE. Slice-aware admission control and "
     "differentiated handling let operators isolate services such as eMBB, URLLC and "
     "mMTC on the same physical infrastructure."),

    # ---- TS 23.501 5G System architecture ----
    ("TS 23.501", "4.2", "5G system architecture and network functions",
     "The 5G core is a service-based architecture composed of network functions "
     "including the AMF for access and mobility management, the SMF for session "
     "management, the UPF for user-plane forwarding, the PCF for policy, the UDM for "
     "subscription data and the NRF for NF discovery. Functions expose services over "
     "HTTP/2 on the service-based interface."),
    ("TS 23.501", "5.6", "PDU sessions and PDU session types",
     "A PDU session provides connectivity between the UE and a data network "
     "identified by a DNN. Supported PDU session types are IPv4, IPv6, IPv4v6, "
     "Ethernet and Unstructured. A UE can have multiple simultaneous PDU sessions, "
     "possibly to different data networks and served by different UPFs and SMFs."),
    ("TS 23.501", "5.7", "QoS model, QoS flows and 5QI",
     "The 5G QoS model is based on QoS flows, the finest granularity of QoS "
     "differentiation in a PDU session. Each QoS flow is identified by a QFI and "
     "characterised by a 5QI that maps to resource type, priority level, packet "
     "delay budget and packet error rate. GBR and non-GBR flows are supported, and "
     "reflective QoS lets the UE derive uplink mapping from downlink traffic."),
    ("TS 23.501", "5.15", "Network slicing and S-NSSAI",
     "A network slice is identified by an S-NSSAI consisting of a slice/service type "
     "and an optional slice differentiator. During registration the UE provides "
     "requested NSSAI and the network returns allowed NSSAI. Slice selection "
     "influences AMF selection and the set of network slice instances the UE may use."),
    ("TS 23.501", "5.3", "Registration management and connection management states",
     "Registration management tracks whether a UE is registered (RM-REGISTERED) or "
     "deregistered (RM-DEREGISTERED) with the network, while connection management "
     "tracks the NAS signalling connection as CM-IDLE or CM-CONNECTED. These states "
     "are maintained independently in the UE and the AMF."),
    ("TS 23.501", "5.8", "User Plane Function and packet forwarding",
     "The UPF anchors PDU sessions, performs packet routing and forwarding, applies "
     "QoS enforcement and gates, generates usage reports for charging and acts as the "
     "point of interconnect to the data network. Uplink classifiers and branching "
     "points enable local traffic breakout and edge computing."),

    # ---- TS 23.502 procedures ----
    ("TS 23.502", "4.3.2", "PDU session establishment procedure",
     "PDU session establishment begins with the UE sending a NAS PDU Session "
     "Establishment Request inside a Service Request or Registration message. The AMF "
     "selects an SMF, the SMF selects a UPF and assigns an IP address, sets up the N4 "
     "session over PFCP, and the session is completed when radio and N3 tunnel "
     "resources are configured."),
    ("TS 23.502", "4.2.3", "Service request procedure",
     "The service request procedure is used by a UE in CM-IDLE to re-establish the "
     "NAS signalling connection and user-plane resources, or by the network via "
     "paging. It restores the N3 tunnels for the requested PDU sessions and moves the "
     "UE to CM-CONNECTED."),
    ("TS 23.502", "4.9.1", "Xn based handover procedure",
     "In an Xn based handover the source and target gNBs coordinate directly over the "
     "Xn interface without core-network involvement for the handover decision. The "
     "source gNB forwards user data to the target, and after the UE accesses the "
     "target cell a path switch procedure updates the UPF downlink tunnel via the AMF "
     "and SMF."),

    # ---- TS 24.501 NAS ----
    ("TS 24.501", "5.5.1", "5GMM registration procedure",
     "The 5GS mobility management registration procedure registers the UE for 5GS "
     "services. The UE sends a Registration Request carrying the 5GS registration "
     "type, the 5GS mobile identity (SUCI or 5G-GUTI) and UE capabilities. The AMF "
     "may run authentication and security, then returns a Registration Accept with "
     "the allowed NSSAI and a new 5G-GUTI."),
    ("TS 24.501", "5.4.1", "Primary authentication and key agreement",
     "Primary authentication in 5G uses either 5G-AKA or EAP-AKA prime. The AMF, "
     "acting as SEAF, obtains authentication vectors from the AUSF and UDM. In 5G-AKA "
     "the UE verifies the network with AUTN and returns RES star, which the network "
     "compares against the expected value to authenticate the subscriber."),
    ("TS 24.501", "5.4.2", "Security mode control and NAS security",
     "The NAS security mode control procedure activates NAS integrity protection and "
     "ciphering. The AMF sends a Security Mode Command selecting the 5G integrity and "
     "encryption algorithms, protected with the new NAS security context, and the UE "
     "replies with a Security Mode Complete once it has verified the message."),

    # ---- TS 33.501 Security ----
    ("TS 33.501", "6.1", "5G security architecture and key hierarchy",
     "5G security defines a key hierarchy rooted in the long-term key K stored in the "
     "USIM and the ARPF. Authentication derives KAUSF and KSEAF, from which KAMF and "
     "then the NAS keys and the KgNB access-stratum keys are derived. Keys are "
     "separated per algorithm and refreshed at mobility and state transitions."),
    ("TS 33.501", "6.12", "Subscriber privacy: SUPI, SUCI and concealment",
     "To protect subscriber privacy the permanent identifier SUPI is never sent in "
     "clear over the air. Instead the UE computes a SUCI by encrypting the SUPI with "
     "the home network public key using the ECIES scheme. The UDM/SIDF de-conceals "
     "the SUCI to recover the SUPI at the home network."),
    ("TS 33.501", "6.7", "Access stratum security and key derivation",
     "Access-stratum security protects RRC and user-plane traffic between the UE and "
     "gNB. KgNB is derived from KAMF and used to derive RRC integrity, RRC ciphering "
     "and UP ciphering keys. Horizontal and vertical key derivation with the NH and "
     "NCC parameters provide key freshness during handovers."),

    # ---- Physical layer TS 38.211/212/213/214 ----
    ("TS 38.211", "7.4.3", "Synchronization Signal Block (SSB)",
     "The SS/PBCH block, or SSB, comprises the primary and secondary synchronization "
     "signals and the PBCH. It occupies four OFDM symbols and 240 subcarriers and is "
     "transmitted in beam-swept bursts. The UE uses PSS and SSS to acquire cell "
     "identity and timing, and decodes the PBCH to obtain the master information "
     "block."),
    ("TS 38.213", "8.1", "Random access and PRACH procedure",
     "The random access procedure lets the UE obtain uplink synchronization and a "
     "C-RNTI. In contention-based access the UE transmits a PRACH preamble (Msg1), "
     "receives a random access response (Msg2), sends Msg3 on PUSCH and resolves "
     "contention with Msg4. NR also defines a two-step RACH that combines Msg1 and "
     "Msg3 into MsgA."),
    ("TS 38.213", "10.1", "PDCCH monitoring and search spaces",
     "The UE monitors the PDCCH for downlink control information in configured search "
     "space sets associated with control resource sets (CORESETs). Common and "
     "UE-specific search spaces define the candidates and monitoring occasions; DCI "
     "formats carry scheduling grants, and the CRC is scrambled with an RNTI to "
     "address the intended UE."),
    ("TS 38.214", "5.1.3", "Downlink modulation, coding and MCS",
     "Downlink data on the PDSCH is adapted through a modulation and coding scheme "
     "index that selects the modulation order (QPSK up to 256QAM) and the code rate. "
     "The gNB chooses the MCS from CSI feedback such as the CQI so that the estimated "
     "block error rate stays around the target of ten percent."),
    ("TS 38.321", "5.4.1", "HARQ operation in the MAC layer",
     "Hybrid ARQ combines forward error correction with retransmissions. NR uses "
     "multiple stop-and-wait HARQ processes so transmissions can continue while "
     "awaiting acknowledgements. Incremental redundancy retransmits different coded "
     "bits, and the receiver soft-combines them to improve the chance of successful "
     "decoding."),
    ("TS 38.321", "5.7", "Discontinuous reception (DRX) in connected mode",
     "Connected-mode DRX lets the UE switch off its receiver during inactivity to "
     "save power. A DRX cycle defines on-duration and inactivity timers during which "
     "the UE monitors PDCCH; outside the active time the UE may skip monitoring. "
     "Short and long DRX cycles trade responsiveness against power consumption."),

    # ---- Paging / mobility / LTE contrast ----
    ("TS 38.304", "7.1", "Paging and DRX in idle and inactive mode",
     "In RRC_IDLE and RRC_INACTIVE the UE monitors paging occasions determined by its "
     "identity and the configured paging DRX cycle. The network pages the UE to "
     "deliver mobile-terminated data or signalling; RAN paging is used to reach a UE "
     "in RRC_INACTIVE within its RAN notification area."),
    ("TS 38.304", "5.2.3", "Cell selection and reselection criteria",
     "Cell selection uses the S-criterion, requiring the measured cell quality to "
     "exceed minimum RSRP and RSRQ thresholds. Cell reselection ranks cells with the "
     "R-criterion using RSRP, cell offsets and hysteresis, with priority-based "
     "reselection across frequencies to keep the UE camped on a suitable cell."),
    ("TS 36.300", "10.1.2", "LTE X2 handover for comparison",
     "In LTE the X2 handover is performed directly between eNodeBs over the X2 "
     "interface, analogous to the NR Xn handover. The source eNB sends a Handover "
     "Request, forwards buffered data, and the MME updates the S-GW bearer path after "
     "the UE completes access to the target eNB."),
    ("TS 36.331", "5.3.1", "LTE RRC connection and states",
     "LTE defines only two RRC states, RRC_IDLE and RRC_CONNECTED, unlike NR which "
     "adds RRC_INACTIVE. LTE RRC connection setup establishes SRB1 with an "
     "RRCConnectionSetup message, and dedicated configuration is delivered through "
     "RRCConnectionReconfiguration."),

    # ---- Transport / interfaces ----
    ("TS 38.401", "6.1", "NG-RAN architecture: gNB, CU and DU split",
     "The NG-RAN consists of gNBs connected to the 5G core over the NG interface and "
     "to each other over Xn. A gNB can be split into a central unit and one or more "
     "distributed units connected by the F1 interface, with the CU further split into "
     "control-plane and user-plane parts over the E1 interface."),
    ("TS 38.413", "8.3", "NGAP UE context and PDU session resource setup",
     "The NG Application Protocol on the N2 interface carries UE-associated signalling "
     "between the gNB and the AMF. Procedures include Initial Context Setup, PDU "
     "Session Resource Setup and UE Context Release, transporting NAS messages and the "
     "QoS parameters needed to configure radio and transport bearers."),
    ("TS 29.244", "7.5", "PFCP and the N4 interface",
     "The Packet Forwarding Control Protocol on the N4 interface lets the SMF program "
     "the UPF. Using Packet Detection Rules, Forwarding Action Rules, QoS Enforcement "
     "Rules and Usage Reporting Rules, the SMF controls how the UPF detects, forwards, "
     "polices and reports on user-plane packets for each PDU session."),

    # ---- URLLC / positioning / IMS ----
    ("TS 23.501", "5.33", "Ultra-reliable low-latency communication (URLLC)",
     "URLLC targets stringent reliability and latency, for example a packet error "
     "rate of ten to the minus five with a one millisecond user-plane latency. "
     "Supporting features include short transmission durations, grant-free uplink, "
     "PDCP duplication over dual connectivity and conservative MCS selection to raise "
     "reliability."),
    ("TS 38.305", "4.1", "NR positioning methods",
     "NR positioning supports downlink and uplink methods including DL-TDOA, UL-TDOA, "
     "multi-RTT and DL-AoD, using positioning reference signals and sounding reference "
     "signals. The Location Management Function computes the UE location from the "
     "measurements collected over the LTE Positioning Protocol."),
    ("TS 23.228", "4.2", "IMS architecture and session control",
     "The IP Multimedia Subsystem provides SIP-based session control for services "
     "such as voice over 5G. Core elements are the P-CSCF, I-CSCF and S-CSCF proxies "
     "and the HSS. Registration binds the user's public identity to a contact address, "
     "and the S-CSCF applies service triggers during session setup."),
]


QUERIES = [
    ("What connection modes can a 5G phone be in, and which one keeps its stored "
     "context while dropping the air-interface link?",
     ["TS 38.331#4.2.1"]),
    ("How does a handset first get a signalling link and move into connected mode "
     "at the base station?",
     ["TS 38.331#5.3.3"]),
    ("Which control-plane message tells a device to switch to a new cell during a "
     "mobility event?",
     ["TS 38.331#5.3.5", "TS 23.502#4.9.1"]),
    ("When does a device tell the network that a neighbour has become stronger than "
     "the current serving cell?",
     ["TS 38.331#5.5.4"]),
    ("What recovery does a device attempt after losing its radio link?",
     ["TS 38.331#5.3.7"]),
    ("Why does using wider tone spacing give shorter time slots and lower delay on "
     "the air interface?",
     ["TS 38.300#5.1"]),
    ("How can operating on a narrower slice of the carrier help a device use less "
     "battery?",
     ["TS 38.300#6.10"]),
    ("What does a device do when its serving antenna direction is suddenly lost?",
     ["TS 38.300#9.2.6"]),
    ("How can one physical network be divided so that low-latency and massive-IoT "
     "services stay isolated from each other?",
     ["TS 38.300#16.1", "TS 23.501#5.15"]),
    ("Which building blocks of the 5G core handle mobility, sessions, user traffic "
     "forwarding and subscriber data?",
     ["TS 23.501#4.2"]),
    ("What kinds of connectivity can be set up to a data network, for example raw "
     "layer-2 frames or unstructured payloads?",
     ["TS 23.501#5.6"]),
    ("How does 5G differentiate traffic within one connection using flow "
     "identifiers and priority classes?",
     ["TS 23.501#5.7"]),
    ("How does a device ask the network for a particular slice of service during "
     "sign-on?",
     ["TS 23.501#5.15"]),
    ("How does the core separately track whether a device is signed up versus "
     "whether it currently has an active signalling link?",
     ["TS 23.501#5.3"]),
    ("Which core element actually routes and polices the user's data packets toward "
     "the internet?",
     ["TS 23.501#5.8"]),
    ("Walk through how a data connection to the internet is set up, including "
     "picking a session manager and assigning an address.",
     ["TS 23.502#4.3.2"]),
    ("How does an idle device get its data tunnels back, whether it wakes itself or "
     "the network nudges it?",
     ["TS 23.502#4.2.3"]),
    ("How do two neighbouring base stations move a device between themselves and "
     "then redirect the downlink traffic path?",
     ["TS 23.502#4.9.1"]),
    ("When a device joins the network, what identifier does it present and what "
     "does it get back?",
     ["TS 24.501#5.5.1"]),
    ("How does a device confirm it is really talking to the genuine operator during "
     "sign-on, and how does the network verify the SIM?",
     ["TS 24.501#5.4.1", "TS 33.501#6.1"]),
    ("Which step turns on ciphering and integrity protection for non-access-stratum "
     "signalling?",
     ["TS 24.501#5.4.2"]),
    ("How are the encryption keys for the radio link derived from the master SIM "
     "secret in 5G?",
     ["TS 33.501#6.1", "TS 33.501#6.7"]),
    ("How is a subscriber's permanent identity kept hidden from eavesdroppers over "
     "the air?",
     ["TS 33.501#6.12"]),
    ("Which broadcast signals let a phone find a cell's identity and timing when it "
     "first powers on?",
     ["TS 38.211#7.4.3"]),
    ("How does a device that is not yet time-aligned get onto the uplink and obtain "
     "a temporary identifier?",
     ["TS 38.213#8.1"]),
    ("Where does a device look for its scheduling grants on the downlink control "
     "channel?",
     ["TS 38.213#10.1"]),
    ("How does the base station pick the constellation and code rate for downlink "
     "data based on channel feedback?",
     ["TS 38.214#5.1.3"]),
    ("How do retransmissions that send extra parity bits get combined at the "
     "receiver to decode a failed block?",
     ["TS 38.321#5.4.1"]),
    ("How can a connected device switch its receiver off periodically to save "
     "power?",
     ["TS 38.321#5.7"]),
    ("How does the network reach an idle or dormant device that has downlink data "
     "waiting?",
     ["TS 38.304#7.1"]),
    ("On what signal-strength rules does an idle device decide which cell to camp "
     "on?",
     ["TS 38.304#5.2.3"]),
    ("How is a base station split into a central part and remote radio units, and "
     "which links connect them?",
     ["TS 38.401#6.1"]),
    ("How does the session manager tell the user-plane node how to detect, forward "
     "and count packets?",
     ["TS 29.244#7.5"]),
    ("Which service class aims for extreme reliability and about one millisecond of "
     "delay, and how is that met?",
     ["TS 23.501#5.33"]),
    ("How can a 5G network estimate exactly where a device is located?",
     ["TS 38.305#4.1"]),
    ("How does a device with no active signalling link restore its user-plane "
     "tunnels?",
     ["TS 23.502#4.2.3"]),
]


def build():
    corpus_path = os.path.join(HERE, "corpus.jsonl")
    queries_path = os.path.join(HERE, "queries.jsonl")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for spec, section, title, text in PASSAGES:
            doc = {
                "id": f"{spec}#{section}",
                "spec": spec,
                "section": section,
                "title": title,
                "text": text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # sanity check: every relevant id must exist in the corpus
    ids = {f"{spec}#{section}" for spec, section, _, _ in PASSAGES}
    with open(queries_path, "w", encoding="utf-8") as f:
        for i, (query, relevant) in enumerate(QUERIES):
            for r in relevant:
                assert r in ids, f"Query {i} references unknown doc id: {r}"
            f.write(json.dumps(
                {"qid": f"q{i:02d}", "query": query, "relevant": relevant},
                ensure_ascii=False) + "\n")

    print(f"Wrote {len(PASSAGES)} passages -> {corpus_path}")
    print(f"Wrote {len(QUERIES)} queries  -> {queries_path}")


if __name__ == "__main__":
    build()
