from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from canon.config import load_settings
from canon.product import report_io


DEFAULT_BENCHMARK_ID = "disagreement_preservation_publishable_v1"
DEFAULT_OUTPUT_PATH = Path("gold") / "disagreement_preservation_publishable.json"


CASE_SPECS: list[dict[str, str]] = [
    {
        "id": "grid_data_center_peak_load",
        "question": "What does the evidence say about data center load and grid stress?",
        "domain": "energy_policy",
        "subject": "grid planning",
        "driver": "data center load",
        "outcome": "substation peak demand",
        "period": "2025",
        "qualifier": "interconnection timing and cooling schedules",
        "neutral_topic": "regional arts funding",
    },
    {
        "id": "hospital_ai_triage_wait_times",
        "question": "What does the evidence say about AI triage and emergency wait times?",
        "domain": "health_operations",
        "subject": "emergency department operations",
        "driver": "AI triage routing",
        "outcome": "median emergency wait times",
        "period": "the winter pilot",
        "qualifier": "night-shift staffing and ambulance diversion",
        "neutral_topic": "cafeteria renovation",
    },
    {
        "id": "remote_monitoring_readmissions",
        "question": "What does the evidence say about remote monitoring and readmissions?",
        "domain": "health_services",
        "subject": "cardiology follow-up",
        "driver": "remote monitoring enrollment",
        "outcome": "thirty-day readmissions",
        "period": "the 2024 rollout",
        "qualifier": "patient broadband access and nurse call capacity",
        "neutral_topic": "parking permits",
    },
    {
        "id": "school_tutoring_math_scores",
        "question": "What does the evidence say about tutoring and math scores?",
        "domain": "education_policy",
        "subject": "middle-school instruction",
        "driver": "high-dosage tutoring",
        "outcome": "eighth-grade math scores",
        "period": "the spring term",
        "qualifier": "attendance consistency and tutor training",
        "neutral_topic": "library carpet replacement",
    },
    {
        "id": "transit_bus_lanes_commute_times",
        "question": "What does the evidence say about bus lanes and commute times?",
        "domain": "transportation",
        "subject": "urban transit operations",
        "driver": "dedicated bus lanes",
        "outcome": "morning commute times",
        "period": "the six-month corridor pilot",
        "qualifier": "signal priority and curb enforcement",
        "neutral_topic": "museum visitor badges",
    },
    {
        "id": "ev_charging_feeder_overloads",
        "question": "What does the evidence say about EV charging and feeder overloads?",
        "domain": "energy_policy",
        "subject": "distribution-grid operations",
        "driver": "evening EV charging",
        "outcome": "feeder overload events",
        "period": "summer 2025",
        "qualifier": "managed charging adoption and transformer age",
        "neutral_topic": "community garden grants",
    },
    {
        "id": "wildfire_cameras_detection_time",
        "question": "What does the evidence say about wildfire cameras and detection time?",
        "domain": "public_safety",
        "subject": "wildfire response",
        "driver": "camera-based smoke detection",
        "outcome": "initial detection time",
        "period": "the dry-season deployment",
        "qualifier": "ridge visibility and dispatcher review latency",
        "neutral_topic": "recreation trail signage",
    },
    {
        "id": "fertilizer_buffer_zones_nitrate_runoff",
        "question": "What does the evidence say about buffer zones and nitrate runoff?",
        "domain": "water_environment",
        "subject": "watershed management",
        "driver": "fertilizer buffer zones",
        "outcome": "nitrate runoff concentrations",
        "period": "the planting season",
        "qualifier": "rainfall timing and field slope",
        "neutral_topic": "sports field reservations",
    },
    {
        "id": "telehealth_no_show_rates",
        "question": "What does the evidence say about telehealth and appointment no-shows?",
        "domain": "health_operations",
        "subject": "outpatient scheduling",
        "driver": "telehealth visit availability",
        "outcome": "appointment no-show rates",
        "period": "the 2024 clinic cycle",
        "qualifier": "language support and reminder workflows",
        "neutral_topic": "wall paint procurement",
    },
    {
        "id": "body_cameras_complaint_rates",
        "question": "What does the evidence say about body cameras and complaint rates?",
        "domain": "public_safety",
        "subject": "police oversight",
        "driver": "body camera activation",
        "outcome": "civilian complaint rates",
        "period": "the first policy year",
        "qualifier": "supervisor audits and missing footage rules",
        "neutral_topic": "holiday parade logistics",
    },
    {
        "id": "zoning_reform_rent_growth",
        "question": "What does the evidence say about zoning reform and rent growth?",
        "domain": "housing_policy",
        "subject": "rental housing markets",
        "driver": "zoning reform",
        "outcome": "median rent growth",
        "period": "the first two years",
        "qualifier": "permit processing and interest rates",
        "neutral_topic": "streetlight bulb inventory",
    },
    {
        "id": "heat_pumps_winter_peak_demand",
        "question": "What does the evidence say about heat pumps and winter peak demand?",
        "domain": "energy_policy",
        "subject": "residential electrification",
        "driver": "heat pump adoption",
        "outcome": "winter peak demand",
        "period": "January cold snaps",
        "qualifier": "building insulation and backup heat use",
        "neutral_topic": "farmers market vendor fees",
    },
    {
        "id": "nutrition_labels_sugar_purchases",
        "question": "What does the evidence say about nutrition labels and sugar purchases?",
        "domain": "public_health",
        "subject": "retail food policy",
        "driver": "front-of-package nutrition labels",
        "outcome": "sugar-sweetened purchases",
        "period": "the store trial",
        "qualifier": "price promotions and shelf placement",
        "neutral_topic": "delivery dock schedules",
    },
    {
        "id": "cash_transfers_food_security",
        "question": "What does the evidence say about cash transfers and food security?",
        "domain": "social_policy",
        "subject": "household assistance",
        "driver": "monthly cash transfers",
        "outcome": "food security scores",
        "period": "the benefit expansion",
        "qualifier": "rent burden and local food prices",
        "neutral_topic": "office furniture auctions",
    },
    {
        "id": "traffic_calming_crash_rates",
        "question": "What does the evidence say about traffic calming and crash rates?",
        "domain": "transportation",
        "subject": "street safety",
        "driver": "traffic calming installations",
        "outcome": "reported crash rates",
        "period": "the first twelve months",
        "qualifier": "intersection design and enforcement intensity",
        "neutral_topic": "concert venue seating",
    },
    {
        "id": "solar_rooftops_feeder_voltage",
        "question": "What does the evidence say about rooftop solar and feeder voltage?",
        "domain": "energy_policy",
        "subject": "distribution-grid planning",
        "driver": "rooftop solar penetration",
        "outcome": "feeder voltage excursions",
        "period": "midday summer hours",
        "qualifier": "inverter settings and feeder topology",
        "neutral_topic": "park bench repair",
    },
    {
        "id": "nurse_staffing_patient_falls",
        "question": "What does the evidence say about nurse staffing and patient falls?",
        "domain": "health_operations",
        "subject": "inpatient safety",
        "driver": "higher nurse staffing ratios",
        "outcome": "patient fall incidents",
        "period": "the pilot quarter",
        "qualifier": "case mix and bed occupancy",
        "neutral_topic": "gift shop inventory",
    },
    {
        "id": "air_filters_asthma_visits",
        "question": "What does the evidence say about classroom air filters and asthma visits?",
        "domain": "public_health",
        "subject": "school health",
        "driver": "classroom air filter installation",
        "outcome": "student asthma visits",
        "period": "the school year",
        "qualifier": "filter maintenance and outdoor smoke days",
        "neutral_topic": "yearbook photography",
    },
    {
        "id": "recycling_bins_contamination",
        "question": "What does the evidence say about recycling bins and contamination?",
        "domain": "municipal_operations",
        "subject": "waste management",
        "driver": "standardized recycling bins",
        "outcome": "recycling contamination rates",
        "period": "the curbside rollout",
        "qualifier": "resident education and hauler feedback",
        "neutral_topic": "festival banner permits",
    },
    {
        "id": "small_business_grants_closures",
        "question": "What does the evidence say about business grants and closures?",
        "domain": "economic_policy",
        "subject": "small-business recovery",
        "driver": "emergency business grants",
        "outcome": "business closure rates",
        "period": "the recovery year",
        "qualifier": "sector mix and landlord negotiations",
        "neutral_topic": "dog-license renewal forms",
    },
    {
        "id": "language_access_claim_denials",
        "question": "What does the evidence say about language access and claim denials?",
        "domain": "health_equity",
        "subject": "insurance navigation",
        "driver": "language access services",
        "outcome": "claim denial rates",
        "period": "the enrollment season",
        "qualifier": "plan complexity and interpreter availability",
        "neutral_topic": "printer toner orders",
    },
    {
        "id": "stormwater_gardens_flood_calls",
        "question": "What does the evidence say about stormwater gardens and flood calls?",
        "domain": "water_environment",
        "subject": "neighborhood drainage",
        "driver": "stormwater garden installation",
        "outcome": "basement flood service calls",
        "period": "the rainy season",
        "qualifier": "soil saturation and sewer maintenance",
        "neutral_topic": "library summer reading",
    },
    {
        "id": "mental_health_teams_arrests",
        "question": "What does the evidence say about mental health teams and arrests?",
        "domain": "public_safety",
        "subject": "crisis response",
        "driver": "mobile mental health teams",
        "outcome": "behavioral-health arrest rates",
        "period": "the dispatch pilot",
        "qualifier": "call triage and police co-response rules",
        "neutral_topic": "city hall window cleaning",
    },
    {
        "id": "broadband_subsidies_homework_completion",
        "question": "What does the evidence say about broadband subsidies and homework completion?",
        "domain": "education_policy",
        "subject": "student connectivity",
        "driver": "broadband subsidies",
        "outcome": "homework completion rates",
        "period": "the academic year",
        "qualifier": "device access and after-school support",
        "neutral_topic": "gymnasium floor polish",
    },
    {
        "id": "permit_digitization_approval_times",
        "question": "What does the evidence say about permit digitization and approval times?",
        "domain": "municipal_operations",
        "subject": "building permits",
        "driver": "permit digitization",
        "outcome": "approval cycle times",
        "period": "the software transition",
        "qualifier": "staff training and application completeness",
        "neutral_topic": "holiday wreath storage",
    },
    {
        "id": "senior_meal_delivery_er_visits",
        "question": "What does the evidence say about meal delivery and emergency visits?",
        "domain": "health_services",
        "subject": "senior services",
        "driver": "meal delivery expansion",
        "outcome": "emergency department visits",
        "period": "the six-month program",
        "qualifier": "caregiver support and medication management",
        "neutral_topic": "pool pass printing",
    },
    {
        "id": "tree_canopy_surface_temperature",
        "question": "What does the evidence say about tree canopy and surface temperature?",
        "domain": "climate_adaptation",
        "subject": "urban heat mitigation",
        "driver": "tree canopy expansion",
        "outcome": "afternoon surface temperature",
        "period": "summer heat waves",
        "qualifier": "watering schedules and pavement materials",
        "neutral_topic": "public art plaques",
    },
    {
        "id": "fare_free_transit_ridership",
        "question": "What does the evidence say about fare-free transit and ridership?",
        "domain": "transportation",
        "subject": "public transit demand",
        "driver": "fare-free transit",
        "outcome": "weekly ridership",
        "period": "the promotional period",
        "qualifier": "service frequency and downtown events",
        "neutral_topic": "uniform patch design",
    },
    {
        "id": "vaccination_reminders_completion",
        "question": "What does the evidence say about vaccination reminders and completion?",
        "domain": "public_health",
        "subject": "immunization outreach",
        "driver": "text-message vaccination reminders",
        "outcome": "series completion rates",
        "period": "the outreach campaign",
        "qualifier": "phone-number accuracy and clinic hours",
        "neutral_topic": "elevator inspection notices",
    },
    {
        "id": "procurement_prepayment_supplier_bids",
        "question": "What does the evidence say about prepayment terms and supplier bids?",
        "domain": "economic_policy",
        "subject": "public procurement",
        "driver": "prepayment terms",
        "outcome": "small supplier bid participation",
        "period": "the annual procurement cycle",
        "qualifier": "contract size and invoice processing speed",
        "neutral_topic": "lobby plant replacement",
    },
]


def build_disagreement_benchmark() -> dict[str, Any]:
    return {
        "benchmark_id": DEFAULT_BENCHMARK_ID,
        "description": (
            "Target-sized controlled Stage 2 fixture with support, contradiction, "
            "qualification, and neutral distractor evidence for synthesis review scaffolding."
        ),
        "case_count": len(CASE_SPECS),
        "case_design": {
            "per_case_evidence": [
                "supporting evidence",
                "contradicting evidence",
                "qualifying evidence",
                "neutral distractor evidence",
            ],
            "human_review_boundary": (
                "The fixture is deterministic scaffolding for review and regression; "
                "it is not a claim about naturally occurring evidence quality."
            ),
        },
        "cases": [case_from_spec(index + 1, spec) for index, spec in enumerate(CASE_SPECS)],
    }


def case_from_spec(index: int, spec: dict[str, str]) -> dict[str, Any]:
    prefix = f"D{index:02d}"
    subject = spec["subject"]
    driver = spec["driver"]
    outcome = spec["outcome"]
    period = spec["period"]
    qualifier = spec["qualifier"]
    domain = spec["domain"]
    return {
        "id": spec["id"],
        "question": spec["question"],
        "mode": "disagreement_fixture",
        "evidence": [
            evidence_row(
                evidence_id=f"{prefix}-S",
                title=f"{subject.title()} Support Memo",
                source_name=f"{subject.title()} Analyst A",
                source_type="analysis_memo",
                domain=domain,
                cluster_id=f"{spec['id']}:main",
                expected_stance="supports",
                text=(
                    f"The {subject} assessment reports that {driver} increased {outcome} "
                    f"during {period}."
                ),
            ),
            evidence_row(
                evidence_id=f"{prefix}-C",
                title=f"{subject.title()} Audit Challenge",
                source_name=f"{subject.title()} Analyst B",
                source_type="audit_review",
                domain=domain,
                cluster_id=f"{spec['id']}:main",
                expected_stance="contradicts",
                text=(
                    f"The {subject} audit says {driver} did not increase {outcome} "
                    f"during {period} after controls."
                ),
            ),
            evidence_row(
                evidence_id=f"{prefix}-Q",
                title=f"{subject.title()} Boundary Note",
                source_name=f"{subject.title()} Technical Panel",
                source_type="technical_note",
                domain=domain,
                cluster_id=f"{spec['id']}:main",
                expected_stance="qualifies",
                text=(
                    f"Although {driver} increased {outcome} in some observations, the "
                    f"{subject} finding depends on {qualifier} and remains limited."
                ),
            ),
            evidence_row(
                evidence_id=f"{prefix}-N",
                title=f"{subject.title()} Administrative Notice",
                source_name=f"{subject.title()} Clerk",
                source_type="administrative_notice",
                domain="administrative",
                cluster_id=f"{spec['id']}:distractor",
                expected_stance="neutral",
                text=(
                    f"The administrative notice describes {spec['neutral_topic']} for routine "
                    f"operations and contains no analysis of the {subject} outcome."
                ),
            ),
        ],
    }


def evidence_row(
    *,
    evidence_id: str,
    title: str,
    source_name: str,
    source_type: str,
    domain: str,
    cluster_id: str,
    expected_stance: str,
    text: str,
) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "title": title,
        "source_name": source_name,
        "source_type": source_type,
        "domain": domain,
        "cluster_id": cluster_id,
        "expected_stance": expected_stance,
        "text": text,
    }


def write_benchmark(output: Path | None = None) -> Path:
    settings = load_settings()
    path = output or settings.root / DEFAULT_OUTPUT_PATH
    if not path.is_absolute():
        path = settings.root / path
    report_io.write_json(path, build_disagreement_benchmark())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the publishable disagreement fixture.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    path = write_benchmark(Path(args.output))
    print(json.dumps({"status": "written", "path": str(path).replace("\\", "/")}, indent=2))


if __name__ == "__main__":
    main()
