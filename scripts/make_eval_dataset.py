"""Generate a self-contained synthetic RAG evaluation corpus.

Writes synthetic documents (no user uploads) to data/eval_corpus/, chunks
and embeds them through the production path, stores them in a dedicated
Chroma collection, and derives a golden set by locating planted fact
markers inside the stored chunks — so relevance labels stay correct even
if chunking changes.

    venv\\Scripts\\python.exe scripts\\make_eval_dataset.py

Queries are tagged with a difficulty class:
  keyword     – query terms appear (nearly) verbatim in the answer chunk
  paraphrase  – answer is reworded; low token overlap (dense-friendly)
  lexical     – hyphenation / compound variants split the tokens (BM25-hostile)
Facts planted in repeated header blocks land in several chunks and become
multi-relevant queries, mirroring real page-header artifacts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from config import NIM_EMBEDDING_CONFIG
from backend.services.component_registry import ComponentRegistry
from indexing.text_chunker import chunk_document, deterministic_chunk_id
from ingestion.document_processor import DocumentProcessor

EVAL_COLLECTION = "neurax_eval_synthetic_v1"
CORPUS_DIR = PROJECT_ROOT / "data" / "eval_corpus"
GOLDEN_PATH = PROJECT_ROOT / "scripts" / "eval_golden_synthetic.json"
EMBED_BATCH = 32

def harbor_header(section: int) -> str:
    # Section-tagged so headers differ per section: near-duplicate page
    # headers survive exact-dedupe and their shared marker line lands in
    # several chunks (multi-relevant queries), like real letterhead.
    return (
        f"PORT OF NAGAR BAY — HARBOR OPS MEMO // REF: HL-2026-114-S{section}\n"
        "Vessel clearance window: 0600-1800 LT // Duty desk: VHF 14\n"
    )


def lumen_header(section: int) -> str:
    return (
        f"ENTITY: LUMEN VENTURES LLC // REG: LV-77-204931 // DELAWARE\n"
        f"SERIES SEED TERM SHEET // DRAFT 2026-05-11 // PAGE {section} // CONFIDENTIAL\n"
    )

DOCUMENTS = [
    {
        "name": "aurora_station_report",
        "title": "Aurora Station Operations Report Q2 2026",
        "text": """Aurora Station Operations Report — Second Quarter 2026

Overview. Aurora Station hosted 38 personnel across the quarter, down from
41 in Q1 after two scheduled departures. All daily traffic moves over the
primary radio channel AURORA-7 on 121.65 MHz, with 156.800 MHz retained as
the emergency fallback. Power generation averaged 94 percent availability
on the primary diesel array, and the wind turbine contributed a further
11 percent on windy weeks.

Fuel and logistics. Bulk diesel stands at 61 percent of rated capacity
after the April resupply, enough for roughly 140 days of normal operations.
The fuel officer flagged a slow leak at the day-tank manifold; a gasket kit
arrives with the next vessel. Speaking of which, the next resupply vessel
is scheduled to dock on September 3, 2026, weather permitting, carrying
68 tonnes of cargo including the manifold gasket kit, fresh provisions,
and two replacement snowmobiles.

Staffing. Engineering falls under chief engineer Maren Voss, now in her
fourth winter rotation; she has asked for an apprentice mechanic to arrive
no later than October. The medical bay is covered by Dr. Ilya Petrov, who
rotates out in November. Station leadership is reviewing whether a second
doctor is justified for the summer season, when transient science parties
push the population above 50.

Safety. Two near-miss events were logged: a forklift brake fault in the
cargo tunnel and a brief generator overload during a blizzard. Neither
resulted in injury. The quarterly safety review is due with the home
office by July 15, 2026, and the station budget forecast assumes a 4
percent increase in fuel spend for the coming year.""",
    },
    {
        "name": "harbor_logistics_memo",
        "title": "Nagar Bay Harbor Logistics Memo",
        "text": f"""{harbor_header(1)}
Section 1 — Cargo handling.

The NC-4 crane on the north quay is rated for 42 tonnes and passed its
annual load test on May 2. The older NC-2 crane stays limited to 18
tonnes until its wire ropes are replaced; spares are on order with an
expected lead time of nine weeks.

• Container dwell time averaged 3.1 days this month, down from 4.0.
• Reefer plugs on the east apron: 64 available, 51 in use.
• Dangerous-goods paperwork: hazardous manifest requires the duty
  harbormaster's counter-signature before any transfer begins.
• Night shifts now run two gangs instead of three; overtime claims fell
  by 22 percent.

{harbor_header(2)}
Section 2 — Clearance and customs.

Import declarations must reach the customs office no later than 48 hours
before a vessel berths. Late paperwork triggered 11 demurrage disputes in
May, costing the terminal roughly nineteen thousand dollars. The customs
office accepts electronic submissions through the port community system;
paper forms are being phased out by the end of the fiscal year.

{harbor_header(3)}
Section 3 — Vendor notes.

The preferred ship-chandler for the quarter is Velasquez Marine Supplies,
which underbid the incumbent by 7 percent on provisions and fresh water.
The duty desk reminds all masters that VHF channel 14 is monitored around
the clock for berth assignments and tow orders.""",
    },
    {
        "name": "meridian_health_baseline",
        "title": "Meridian District Health Baseline 2026",
        "text": """Meridian District Health Baseline Survey — 2026 edition

Immunization. The spring campaign reached 31,400 children across 118
villages. Measles coverage reached 93.4 percent, up from 89.1 percent last
year, while full-dose polio stands at 91.8 percent. Village-level gaps
remain in the river north quadrant, where three settlements were missed
because boats were unavailable during the flood peak.

Facilities. The district operates one 40-bed hospital and eleven outreach
posts. Cold-chain medicines are stored in the pharmacy refrigerator held
at 4 degrees Celsius, backed by a generator that carried the unit through
two multi-day power outages in March. A second solar-powered cold box is
being procured for the eastern outreach circuit.

Workforce. The district employs 63 nurses and 9 physicians. Retention is
the main concern: four nurses resigned in the past six months, all citing
housing. The health committee has budgeted for six staff quarters to be
renovated before the monsoon.

Nutrition and surveillance. Moderate acute malnutrition was recorded in
4.2 percent of screened under-fives. The weekly epidemiological bulletin
now includes dengue and typhoid reporting; last month it flagged a small
cluster of typhoid cases near the market ward, traced to a shared well
that has since been chlorinated.""",
    },
    {
        "name": "kestrel_flight_ops",
        "title": "Kestrel UAS Flight Operations Specification",
        "text": """KESTREL UAS FLIGHT OPERATIONS SPECIFICATION REV C

Airframe and performance summary
 wingspan of 3.4 meters with a carbon-fiber spar joined at the root
 hardpoints rated at 6 kg total payload split across two wing pylons
 airframe empty mass 9.7 kg max takeoff mass 21.5 kg
 launch by pneumatic rail recovery by belly skid or 30 m arresting line
 flight controller enforces a hard ceiling of 4,800 meters above launch
 true airspeed 26 m/s cruise 34 m/s maximum in clean configuration
 endurance 96 minutes at optimal glide ratio with a light sensor load

Power system
 battery pack uses lithium-ion cells in a 12S2P arrangement 44.4 V nominal
 sustained current draw 18 A at cruise with payload avionics online
 cold-weather kit keeps the pack above 5 degrees C for arctic deployments
 spare packs recharge in 70 minutes on the dual-bay ground charger

Sensors and datalink
 standard fit is a 5-band multispectral camera on a stabilized gimbal
 optional thermal pod replaces the multispectral unit within minutes
 line-of-sight datalink reaches 42 km with the tracking antenna array
 lost-link behavior is a loiter climb then autonomous return to launch

Crewing and limits
 operations require a rated pilot plus one observer for beyond-visual work
 crosswind limit for rail launch is 8 m/s gusts included
 icing flight is prohibited the airframe carries no ice protection""",
    },
    {
        "name": "lumen_ventures_term_sheet",
        "title": "Lumen Ventures Series Seed Term Sheet",
        "text": f"""{lumen_header(1)}
1. Investment terms.

The company and the investors agree on a Series Seed financing with a
valuation cap of 18 million dollars and a 20 percent discount to the next
qualified round. The lead is Cormorant Capital Partners, joined by two
angel syndicates. The round remains open until the earlier of a 2.5
million dollar raise or November 30, 2026.

{lumen_header(2)}
2. Governance.

Investor shall designate one member of the board of directors, which
otherwise consists of two founders and one independent director to be
mutually agreed. Protective provisions apply to new share classes, debt
above 250 thousand dollars, and any change of control. Founder shares
vest over four years with a one-year cliff.

{lumen_header(3)}
3. Information rights and closing.

Monthly financial summaries and an annual budget are due within 45 days
of fiscal year end. The company maintains books with an independent
accountant acceptable to the lead investor. Closing conditions include a
clean cap table, employment agreements for both founders, and the
intellectual property assignment schedule attached as Exhibit B.
Legal fees for the round are split evenly between company and investors.""",
    },
    {
        "name": "silverline_rail_notice",
        "title": "Silverline Rail Maintenance Notice",
        "text": """SILVERLINE RAIL — PASSENGER NOTICE // EFFECTIVE JULY 2026

Engineering work will close the bore between Halden and Fernbrook for
six weekends starting July 11. During the closure, Bracken Tunnel will be
closed to all traffic while crews renew 4.2 kilometers of ballast and
replace 19 crossover timbers. Foot passengers should expect the station
subway at Fernbrook to remain open throughout.

Replacement service. Substitute coaches depart every twenty minutes from
Halden frontage road, calling at Ashfield and Fernbrook, with a journey
time of about 35 minutes. Cyclists may board with bicycles at no extra
charge, but spaces are limited to two per coach. Rail tickets are valid
on the coaches without endorsement.

Fares and compensation. Season-ticket holders may claim a delay-repay
style refund for affected weekends through the website portal; single and
return buyers are not eligible. The operator apologizes for the
disruption and notes that on-time performance should improve by several
points once the renewed section opens. The full engineering calendar is
published each Thursday by 18:00.""",
    },
    {
        "name": "atlas_supply_chain_review",
        "title": "Atlas Components Supply Chain Quarterly Review",
        "text": """Atlas Components — Supply Chain Quarterly Review, Q2 FY2026

Network posture. Ocean freight normalized after a volatile first quarter,
though the cape rerouting added five weeks of transit on the Europe–Asia
lane and pushed quarterly air-freight spend up by 31 percent. Contract
rates for the next two quarters are locked with two carriers, covering
about 70 percent of planned volume.

Supplier base. Concentration remains the top risk: 41 percent of sourced
components trace to a single vendor, Hoshino Precision, whose Kobe plant
is still rebuilding after the spring earthquake. A second source for the
actuator line is being qualified, with an audit visit booked for August.
The vendor scorecard now weights dual-sourcing progress at 15 percent.

Systems. The planning team shipped a multi-modal freight dashboard that
joins ocean, air, and rail telemetry into one lane view; early users
report cut-and-paste work dropping by half. Warehouse automation at the
Dulles distribution center reached phase two, with goods-to-person pickers
live on six aisles. Inventory turns improved from 4.1 to 4.6.

Outlook. Management expects component lead times to ease through the fall
but keeps a 65-day safety stock on long-tail electronic parts. The
quarter's backlog stood at 214 million dollars, book-to-bill 1.08.""",
    },
    {
        "name": "nomad_festival_guide",
        "title": "Nomad Festival Operations Guide 2026",
        "text": """Nomad Festival — Operations Guide (August 2026 edition)

Gates and flow. West Gate admits 12,000 attendees per hour at full
screening strength; East Gate is reserved for crew, artists, and
deliveries before 16:00 each day. Wristband exchanges close 90 minutes
before the headliner. Anyone needing accessibility support can prebook a
companion pass through the access office, and both gates run a fast lane
for pre-registered guests.

Tickets and refunds. Tickets are reimbursable for up to fourteen days
after purchase, minus a 5 percent processing fee, and the window widens
automatically if a headline act cancels. Tier-three pricing ends when the
event sells 80 percent of capacity. Volunteers receive a full weekend
pass plus one meal voucher per shift.

Crew contact. Volunteer guides can reach the coordination office by
email at guides@nomadfestival.org, while production matters go to the
stage managers' radio net on channel 7. Lost property is logged at the
welfare tent near West Gate and held for 30 days after the event.

Safety and noise. The medical post is staffed around the clock with two
paramedics and a doctor on the busiest nights. Sound levels at the
perimeter fence are capped at 65 dB(A) after midnight following the
agreement with the district council.""",
    },
    {
        "name": "obsidian_security_audit",
        "title": "Obsidian Labs Internal Security Audit 2026",
        "text": """Obsidian Labs — Internal Security Audit, first half 2026

Identity. MFA enrollment stands at 87 percent of workforce accounts, with
the remainder concentrated in field contractor roles that lack corporate
devices. Passwordless sign-in via hardware keys is live for all
engineering staff. Dormant accounts are now disabled automatically after
45 days without a login, a control that removed 212 stale identities.

Findings. Three critical findings remain open past their remediation
deadline: an unencrypted backup volume in the legacy file estate, a
service account with a non-rotating key, and an overly permissive IAM
policy on the analytics bucket. The audit committee asked for closure
plans within 30 days. Medium-severity findings fell from 37 to 22.

Detection and response. The security operations center runs a follow-the-
sun rota with analysts in Lisbon and Auckland. Mean time to acknowledge
an alert dropped to 9 minutes, while containment of confirmed incidents
averages 41 minutes. A tabletop exercise simulating ransomware in the
build farm is planned for September.

Vendor posture. Sixty-one vendors completed the security questionnaire;
nine were placed on a watch list. The data protection officer approved
two new sub-processors for the support platform. Annual security training
completion is at 94 percent company-wide.""",
    },
    {
        "name": "quiet_marsh_ecology",
        "title": "Quiet Marsh Field Ecology Survey",
        "text": """Quiet Marsh Field Ecology Survey — Summer 2026

Amphibians. Night call-counts across 14 transects suggest the population
is holding steady: we estimate roughly 4,100 adult northern marsh frogs,
within the confidence band of last summer's estimate. Two natterjack toad
pools near the old sluice dried early this year, and the team moved 300
tadpoles to the deeper scrape by the bird hide.

Invasive plants. Eichhornia crassipes, commonly called water hyacinth,
appeared at three new locations after the June storms, and the warden has
budgeted for manual removal before it seeds. Floating pennywort remains
confined to the drainage ditch along the eastern embankment.

Birds. Bitterns boomed at two territories, matching the recent average.
A pair of cranes nested on the northern island for the first time since
the restoration, fledging one chick. Winter wildfowl counts will resume
in October with the volunteer team.

Hydrology and access. Water levels were held 20 centimeters higher
through spring to aid spawning, at some cost to the grazing marsh
botany. The boardwalk replacement is finished, and the south hide is
open dawn to dusk. Students should book the field laboratory through the
warden's office at least a fortnight in advance.""",
    },
]

# query / marker / class — markers must appear verbatim in exactly one
# document and be long enough to be globally unique.
FACTS = [
    # keyword
    {"query": "What is the primary radio channel of Aurora Station?",
     "marker": "primary radio channel AURORA-7 on 121.65 MHz", "class": "keyword"},
    {"query": "When does the resupply vessel dock at Aurora Station?",
     "marker": "resupply vessel\nis scheduled to dock on September 3, 2026", "class": "keyword"},
    {"query": "What is the lift capacity of the NC-4 crane at Nagar Bay?",
     "marker": "NC-4 crane on the north quay is rated for 42 tonnes", "class": "keyword"},
    {"query": "What measles vaccination rate did the Meridian district reach?",
     "marker": "Measles coverage reached 93.4 percent", "class": "keyword"},
    {"query": "What is the wingspan of the Kestrel airframe?",
     "marker": "wingspan of 3.4 meters", "class": "keyword"},
    {"query": "What valuation cap did Lumen Ventures agree to?",
     "marker": "valuation cap of 18 million dollars", "class": "keyword"},
    {"query": "Which tunnel closes during the Silverline engineering work?",
     "marker": "Bracken Tunnel will be\nclosed", "class": "keyword"},
    {"query": "How many attendees can West Gate admit per hour at the festival?",
     "marker": "West Gate admits 12,000 attendees per hour", "class": "keyword"},
    {"query": "How many critical audit findings are still unresolved?",
     "marker": "Three critical findings remain open", "class": "keyword"},
    # paraphrase (answer reworded; low token overlap)
    {"query": "Who leads the engineering team at Aurora Station?",
     "marker": "chief engineer Maren Voss", "class": "paraphrase"},
    {"query": "How much fuel remains in the station's tanks?",
     "marker": "Bulk diesel stands at 61 percent of rated\ncapacity", "class": "paraphrase"},
    {"query": "Who has to approve hazardous cargo before transfer at the port?",
     "marker": "duty\n  harbormaster's counter-signature", "class": "paraphrase"},
    {"query": "Where are the temperature-sensitive medicines kept at the clinic?",
     "marker": "Cold-chain medicines are stored in the pharmacy refrigerator", "class": "paraphrase"},
    {"query": "How high can the drone fly before its software stops it?",
     "marker": "flight controller enforces a hard ceiling of 4,800 meters", "class": "paraphrase"},
    {"query": "How many board members can the investor appoint?",
     "marker": "Investor shall designate one member of the board of directors", "class": "paraphrase"},
    {"query": "How frequently do the buses replacing the trains run?",
     "marker": "Substitute coaches depart every twenty minutes", "class": "paraphrase"},
    {"query": "What extra travel time did ships incur from the route change?",
     "marker": "cape rerouting added five weeks of transit", "class": "paraphrase"},
    {"query": "How long do festival visitors have to get their money back?",
     "marker": "Tickets are reimbursable for up to fourteen days", "class": "paraphrase"},
    {"query": "What share of employees use two-step sign-in at Obsidian Labs?",
     "marker": "MFA enrollment stands at 87 percent of workforce", "class": "paraphrase"},
    {"query": "How large is the marsh frog population estimated to be?",
     "marker": "roughly 4,100 adult northern marsh frogs", "class": "paraphrase"},
    # lexical (hyphenation / compound / variant spellings split tokens)
    {"query": "What battery chemistry does the Kestrel drone use?",
     "marker": "battery pack uses lithium-ion cells", "class": "lexical"},
    {"query": "What email address should volunteer guides contact?",
     "marker": "email at\nguides@nomadfestival.org", "class": "lexical"},
    {"query": "Which invasive water plant is spreading through the wetland?",
     "marker": "commonly called water hyacinth", "class": "lexical"},
    {"query": "Does the audit cover multi-factor login for contractors?",
     "marker": "field contractor roles that lack corporate\ndevices", "class": "lexical"},
    # multi-relevant (planted in repeated header blocks)
    {"query": "During which hours is the vessel clearance window open at Nagar Bay?",
     "marker": "Vessel clearance window: 0600-1800 LT", "class": "keyword"},
    {"query": "What is the registration number of Lumen Ventures?",
     "marker": "REG: LV-77-204931", "class": "keyword"},
]


def normalize_ws(s: str) -> str:
    return " ".join(s.split())


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    processor = DocumentProcessor()
    registry = ComponentRegistry()
    emb, _, _ = registry.ensure_vector_stack()
    model = emb.text_embedding_model

    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "vector_db"))
    col = client.get_or_create_collection(EVAL_COLLECTION)

    all_rows = []  # (id, text, meta)
    for doc in DOCUMENTS:
        path = CORPUS_DIR / f"{doc['name']}.txt"
        path.write_text(doc["text"].strip() + "\n", encoding="utf-8")
        rel_path = str(path.relative_to(PROJECT_ROOT))
        result = processor.process_file(path)
        result["metadata"] = {"title": doc["title"]}
        chunks = chunk_document(result)
        for i, c in enumerate(chunks):
            cid = deterministic_chunk_id(rel_path, i, model)
            all_rows.append((cid, c["content"], {
                "file_path": rel_path,
                "file_type": "txt",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "title": doc["title"],
                "embedding_model": model,
            }))
        print(f"{doc['name']:28s} {len(chunks):2d} chunks")

    # embed in batches (NIM rejects oversized single requests)
    embeddings = []
    for i in range(0, len(all_rows), EMBED_BATCH):
        batch = [r[1] for r in all_rows[i : i + EMBED_BATCH]]
        vectors = emb.embed_text(batch)
        embeddings.extend(vectors.tolist() if hasattr(vectors, "tolist") else list(vectors))

    col.upsert(
        ids=[r[0] for r in all_rows],
        documents=[r[1] for r in all_rows],
        metadatas=[r[2] for r in all_rows],
        embeddings=embeddings,
    )

    # derive golden labels from markers against stored chunk texts
    chunk_texts = {r[0]: normalize_ws(r[1]) for r in all_rows}
    queries = []
    errors = []
    for fact in FACTS:
        needle = normalize_ws(fact["marker"])
        hits = sorted(cid for cid, text in chunk_texts.items() if needle in text)
        if not hits:
            errors.append(f"marker not found in any chunk: {fact['marker']!r}")
            continue
        queries.append({
            "query": fact["query"],
            "relevant": [cid[:12] for cid in hits],
            "class": fact["class"],
        })
    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)

    golden = {
        "collection": EVAL_COLLECTION,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_dir": str(CORPUS_DIR.relative_to(PROJECT_ROOT)),
        "chunks": len(all_rows),
        "queries": queries,
    }
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"\ncollection {EVAL_COLLECTION}: {len(all_rows)} chunks")
    print(f"golden set: {len(queries)} queries -> {GOLDEN_PATH}")
    from collections import Counter
    for cls, n in Counter(q["class"] for q in queries).items():
        print(f"  {cls:10s} {n}")


if __name__ == "__main__":
    main()
