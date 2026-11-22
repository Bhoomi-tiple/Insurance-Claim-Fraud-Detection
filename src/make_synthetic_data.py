"""FALLBACK ONLY: creates a fake dataset with the same columns as the Kaggle file,
so you can test that the code runs. Do NOT use it for your report/results.

    python -m src.make_synthetic_data          (refuses to overwrite a real file)
    python -m src.make_synthetic_data --force
"""
import sys

import numpy as np
import pandas as pd

from .config import DATA_RAW, SEED


def main(force=False):
    if DATA_RAW.exists() and not force:
        print(f"{DATA_RAW} already exists - not overwriting (use --force).")
        return
    rng = np.random.default_rng(SEED)
    n = 1000
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)

    severity = pick(["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"], [.35, .28, .27, .10])
    hobbies = pick(["sleeping", "reading", "board-games", "bungie-jumping", "base-jumping", "golf", "camping",
                    "dancing", "skydiving", "movies", "hiking", "yachting", "paintball", "chess", "kayaking",
                    "polo", "basketball", "video-games", "cross-fit", "exercise"])
    injury = rng.integers(0, 21000, n)
    prop = rng.integers(0, 25000, n)
    vehicle = rng.integers(1000, 60000, n)
    total = injury + prop + vehicle
    bind = pd.to_datetime("1990-01-01") + pd.to_timedelta(rng.integers(0, 9500, n), unit="D")
    inc = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 59, n), unit="D")

    logit = (-2.7 + 2.9 * (severity == "Major Damage") + 2.2 * np.isin(hobbies, ["chess", "cross-fit"])
             + 0.00002 * (total - 52000) + rng.normal(0, 0.9, n))
    fraud = rng.random(n) < 1 / (1 + np.exp(-logit))

    def with_unknown(opts, frac=0.25):
        v = rng.choice(opts, n).astype(object)
        v[rng.random(n) < frac] = "?"
        return v

    df = pd.DataFrame({
        "months_as_customer": rng.integers(0, 480, n), "age": rng.integers(19, 65, n),
        "policy_number": rng.integers(100000, 999999, n), "policy_bind_date": bind.strftime("%Y-%m-%d"),
        "policy_state": pick(["OH", "IL", "IN"]), "policy_csl": pick(["250/500", "100/300", "500/1000"]),
        "policy_deductable": pick([500, 1000, 2000]), "policy_annual_premium": rng.normal(1250, 250, n).round(2),
        "umbrella_limit": pick([0, 0, 0, 0, 2000000, 4000000, 6000000]),
        "insured_zip": rng.integers(430000, 620000, n), "insured_sex": pick(["MALE", "FEMALE"]),
        "insured_education_level": pick(["MD", "PhD", "Associate", "Masters", "High School", "College", "JD"]),
        "insured_occupation": pick(["tech-support", "craft-repair", "other-service", "sales", "machine-op-inspct",
                                    "armed-forces", "exec-managerial", "prof-specialty", "adm-clerical",
                                    "transport-moving", "handlers-cleaners", "farming-fishing",
                                    "protective-serv", "priv-house-serv"]),
        "insured_hobbies": hobbies,
        "insured_relationship": pick(["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"]),
        "capital-gains": rng.choice([0, 0, 35000, 50000, 68000], n), "capital-loss": rng.choice([0, 0, -30000, -50000, -70000], n),
        "incident_date": inc.strftime("%Y-%m-%d"),
        "incident_type": pick(["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"]),
        "collision_type": with_unknown(["Side Collision", "Rear Collision", "Front Collision"]),
        "incident_severity": severity,
        "authorities_contacted": pick(["Police", "None", "Fire", "Other", "Ambulance"]),
        "incident_state": pick(["SC", "VA", "NY", "OH", "WV", "NC", "PA"]),
        "incident_city": pick(["Columbus", "Riverwood", "Arlington", "Springfield", "Hillsdale", "Northbend", "Northbrook"]),
        "incident_location": [f"{a} Main St" for a in rng.integers(1000, 9999, n)],
        "incident_hour_of_the_day": rng.integers(0, 24, n), "number_of_vehicles_involved": rng.integers(1, 5, n),
        "property_damage": with_unknown(["YES", "NO"], .35), "bodily_injuries": rng.integers(0, 3, n),
        "witnesses": rng.integers(0, 4, n), "police_report_available": with_unknown(["YES", "NO"], .34),
        "total_claim_amount": total, "injury_claim": injury, "property_claim": prop, "vehicle_claim": vehicle,
        "auto_make": pick(["Saab", "Dodge", "Suburu", "Nissan", "Chevrolet", "BMW", "Ford", "Toyota", "Audi",
                           "Accura", "Volkswagen", "Jeep", "Mercedes", "Honda"]),
        "auto_model": pick(["92x", "E400", "RAM", "Tahoe", "Camry", "Civic", "Wrangler", "X5", "A3", "Jetta"]),
        "auto_year": rng.integers(1995, 2016, n),
        "fraud_reported": np.where(fraud, "Y", "N"), "_c39": np.nan,
    })
    DATA_RAW.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_RAW, index=False)
    print(f"Synthetic data written to {DATA_RAW}  (fraud rate {fraud.mean():.1%}). TEST USE ONLY.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
