from __future__ import annotations

from pathlib import Path
import site
import sys
import warnings
import argparse


USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Mean of empty slice")


NUMERIC_HINTS = {
    "Age": ["age", "studentage"],
    "Attendance_Rate": ["attendancerate", "attendance", "attendancepercentage"],
    "Study_Hours": ["studyhours", "hoursstudied", "studytime"],
    "Previous_Academic_Performance": ["previousacademicperformance", "previousperformance", "academicperformance"],
    "Class_Participation": ["classparticipation", "participation", "classengagement"],
}

CATEGORICAL_HINTS = {
    "Gender": ["gender", "sex"],
    "Grade_Level": ["gradelevel", "grade", "classlevel"],
    "Parental_Education_Level": ["parentaleducationlevel", "parenteducation", "parentaleducation"],
    "Parental_Involvement": ["parentalinvolvement", "parentinvolvement"],
    "Extracurricular_Activities": ["extracurricularactivities", "extracurricular", "activities"],
    "Socioeconomic_Status": ["socioeconomicstatus", "socioeconomic", "economicstatus"],
    "Health_Status": ["healthstatus", "health", "wellbeingstatus"],
}

PROFILE_LIBRARY = [
    {
        "name": "High Achiever",
        "description": "Strong attendance, prior performance, and participation suggest a student who responds well to challenge.",
        "strategies": [
            "Give enrichment tasks.",
            "Use peer mentoring roles.",
            "Offer project-based extension work.",
        ],
    },
    {
        "name": "Independent Scholar",
        "description": "High study commitment suggests a learner who works well with autonomy and self-paced tasks.",
        "strategies": [
            "Use self-paced study plans.",
            "Assign reflection journals.",
            "Offer optional deep-dive assignments.",
        ],
    },
    {
        "name": "Collaborative Explorer",
        "description": "Higher participation points to a learner who benefits from social, discussion-led, and applied activities.",
        "strategies": [
            "Use group discussions.",
            "Assign team-based tasks.",
            "Connect lessons to practical activities.",
        ],
    },
    {
        "name": "Support-Seeking Learner",
        "description": "Lower attendance or academic history suggests the need for structure, close follow-up, and guided support.",
        "strategies": [
            "Break learning into small milestones.",
            "Use frequent low-stakes checks.",
            "Pair with tutoring support.",
        ],
    },
    {
        "name": "Resilient Improver",
        "description": "Moderate outcomes with effort suggest growing potential that can improve quickly with encouragement.",
        "strategies": [
            "Track weekly goals.",
            "Celebrate small improvements.",
            "Blend scaffolding with independent practice.",
        ],
    },
    {
        "name": "Wellbeing-Focused Learner",
        "description": "Inconsistent engagement may reflect routine, stress, or access challenges rather than low ability.",
        "strategies": [
            "Provide concise support materials.",
            "Use flexible review support.",
            "Follow up on engagement regularly.",
        ],
    },
]


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def detect_columns(df: pd.DataFrame, hints: dict[str, list[str]]) -> dict[str, str]:
    normalized = {normalize(column): column for column in df.columns}
    detected: dict[str, str] = {}

    for canonical, synonyms in hints.items():
        for synonym in synonyms:
            if synonym in normalized:
                detected[canonical] = normalized[synonym]
                break
        if canonical in detected:
            continue
        for normalized_name, original_name in normalized.items():
            if any(synonym in normalized_name or normalized_name in synonym for synonym in synonyms):
                detected[canonical] = original_name
                break

    return detected


def find_dataset() -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        # root / "Old_data.xlsx",
        # root / "student_performance.csv",
        # root / "Student_Performance.csv",
        # root / "student_performance.xlsx",
        root / "Student_Performance.xlsx",
        Path(__file__).resolve().parent / "Student_Performance.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Dataset not found in project root.")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, LabelEncoder], StandardScaler]:
    numeric_detected = detect_columns(df, NUMERIC_HINTS)
    categorical_detected = detect_columns(df, CATEGORICAL_HINTS)

    numeric_columns = list(numeric_detected.values())
    categorical_columns = list(categorical_detected.values())

    if not numeric_columns:
        raise ValueError("No numeric columns were detected for clustering.")

    feature_df = pd.DataFrame(index=df.index)
    encoders: dict[str, LabelEncoder] = {}

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = series.median()
        fill_value = float(median) if pd.notna(median) else 0.0
        feature_df[column] = series.fillna(fill_value)

    for column in categorical_columns:
        series = df[column].astype("string").str.strip()
        series = series.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA, "NaN": pd.NA})
        mode = series.mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        encoder = LabelEncoder()
        feature_df[column] = encoder.fit_transform(series.fillna(fill_value).astype(str))
        encoders[column] = encoder

    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print("Detected numeric columns:", numeric_columns)
    print("Detected categorical columns:", categorical_columns)
    print("Feature matrix shape:", feature_df.shape)
    print("Remaining NaN count:", int(np.isnan(scaled).sum()))

    return feature_df, scaled, encoders, scaler


def score_column(row: pd.Series, columns: list[str], keyword: str) -> float:
    match = next((column for column in columns if keyword in normalize(column)), None)
    return float(row.get(match, 0.0)) if match else 0.0


def assign_profiles(feature_df: pd.DataFrame, numeric_columns: list[str], labels: np.ndarray) -> dict[int, dict[str, object]]:
    cluster_summary = feature_df[numeric_columns].copy()
    cluster_summary["Cluster"] = labels
    cluster_summary = cluster_summary.groupby("Cluster").mean(numeric_only=True)

    normalized_summary = cluster_summary.copy()
    for column in normalized_summary.columns:
        std = normalized_summary[column].std()
        if std and not np.isnan(std):
            normalized_summary[column] = (normalized_summary[column] - normalized_summary[column].mean()) / std
        else:
            normalized_summary[column] = 0.0

    available_profiles = {profile["name"]: profile for profile in PROFILE_LIBRARY}
    ordered_names = [profile["name"] for profile in PROFILE_LIBRARY]
    used_names: set[str] = set()
    profile_map: dict[int, dict[str, object]] = {}

    for cluster_id in normalized_summary.index:
        row = normalized_summary.loc[cluster_id]
        profile_scores = {
            "High Achiever": score_column(row, numeric_columns, "academic")
            + score_column(row, numeric_columns, "attendance")
            + score_column(row, numeric_columns, "participation"),
            "Independent Scholar": score_column(row, numeric_columns, "study")
            + 0.5 * score_column(row, numeric_columns, "age"),
            "Collaborative Explorer": score_column(row, numeric_columns, "participation"),
            "Support-Seeking Learner": -(
                score_column(row, numeric_columns, "attendance")
                + score_column(row, numeric_columns, "academic")
            ),
            "Resilient Improver": score_column(row, numeric_columns, "study")
            - 0.5 * score_column(row, numeric_columns, "academic"),
            "Wellbeing-Focused Learner": -0.5 * score_column(row, numeric_columns, "attendance"),
        }

        ranked_profiles = sorted(profile_scores.items(), key=lambda item: item[1], reverse=True)
        chosen_name = next((name for name, _ in ranked_profiles if name not in used_names), ranked_profiles[0][0])
        used_names.add(chosen_name)

        count = int((labels == cluster_id).sum())
        percentage = count / len(labels) * 100
        profile_map[cluster_id] = {
            **available_profiles[chosen_name],
            "count": count,
            "percentage": percentage,
            "means": cluster_summary.loc[cluster_id].to_dict(),
        }

    for cluster_id in cluster_summary.index:
        if cluster_id not in profile_map:
            fallback_name = next(name for name in ordered_names if name not in used_names)
            used_names.add(fallback_name)
            count = int((labels == cluster_id).sum())
            percentage = count / len(labels) * 100
            profile_map[cluster_id] = {
                **available_profiles[fallback_name],
                "count": count,
                "percentage": percentage,
                "means": cluster_summary.loc[cluster_id].to_dict(),
            }

    return profile_map


def print_cluster_profiles(profiles: dict[int, dict[str, object]], numeric_columns: list[str]) -> None:
    print("\nCluster interpretation:")
    for cluster_id in sorted(profiles):
        profile = profiles[cluster_id]
        print(f"\nCluster {cluster_id}: {profile['name']}")
        print(f"Students: {profile['count']} ({profile['percentage']:.1f}%)")
        print(f"Description: {profile['description']}")
        print("Mean numeric values:")
        means = profile["means"]
        for column in numeric_columns:
            print(f"  - {column}: {means[column]:.2f}")
        print("Teaching strategies:")
        for strategy in profile["strategies"]:
            print(f"  - {strategy}")


def print_algorithm_summary(model: KMeans, scaled: np.ndarray, labels: np.ndarray, cluster_count: int) -> None:
    silhouette = float(silhouette_score(scaled, labels))
    print("\nAlgorithm used: K-Means")
    print(f"Cluster count: {cluster_count}")
    print("Silhouette score:", round(silhouette, 4))
    print("Inertia:", round(float(model.inertia_), 4))


def train_and_test_prediction(
    scaled: np.ndarray,
    feature_df: pd.DataFrame,
    numeric_columns: list[str],
    cluster_count: int,
) -> None:
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)
    profiles = assign_profiles(feature_df, numeric_columns, labels)

    print("\nTraining successful")
    print_algorithm_summary(model, scaled, labels, cluster_count)
    print("Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index().to_string())
    print_cluster_profiles(profiles, numeric_columns)

    sample_index = 0
    sample_row = feature_df.iloc[[sample_index]]
    sample_scaled = scaled[sample_index].reshape(1, -1)
    predicted_cluster = int(model.predict(sample_scaled)[0])

    print("\nPrediction check successful")
    print(f"Sample row index: {sample_index}")
    print("Sample values:")
    print(sample_row.to_string(index=False))
    print("Predicted cluster:", predicted_cluster)
    print("Predicted learner profile:", profiles[predicted_cluster]["name"])
    print("Profile description:", profiles[predicted_cluster]["description"])
    print("Recommended strategies:")
    for strategy in profiles[predicted_cluster]["strategies"]:
        print(f"  - {strategy}")


def resolve_dataset_path() -> Path:
    parser = argparse.ArgumentParser(description="Prototype clustering pipeline for the student performance dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to a CSV/XLSX dataset. If omitted, the script uses the default project dataset.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of clusters/components to fit. Default: 4",
    )
    args = parser.parse_args()

    if args.data:
        dataset_path = Path(args.data).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        return dataset_path, args.clusters

    return find_dataset(), args.clusters


def main() -> None:
    dataset_path, cluster_count = resolve_dataset_path()
    print("Using dataset:", dataset_path)
    df = load_dataset(dataset_path)
    print("Dataset shape:", df.shape)
    print(f"Model trained on {len(df):,} students")

    feature_df, scaled, _, _ = preprocess(df)
    numeric_columns = list(detect_columns(df, NUMERIC_HINTS).values())
    train_and_test_prediction(scaled, feature_df, numeric_columns, cluster_count)


if __name__ == "__main__":
    main()
