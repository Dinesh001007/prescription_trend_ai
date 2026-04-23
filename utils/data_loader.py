import pandas as pd
import json
import io


def load_file(uploaded_file, filename=None) -> pd.DataFrame:
    """Load CSV or JSON file uploaded via Streamlit or Flask."""
    if filename is None:
        name = uploaded_file.name.lower()
        content = uploaded_file.read()
    else:
        name = filename.lower()
        content = uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
    elif name.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try records format, then orient keys
            if any(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError(f"Unsupported file format: {name}")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_sample_rows(df: pd.DataFrame, n: int = 3) -> list:
    """Return first n rows as list of dicts for LLM context."""
    return df.head(n).fillna("").to_dict(orient="records")


def get_column_types(df: pd.DataFrame) -> dict:
    """Return pandas dtype info for each column."""
    return {col: str(df[col].dtype) for col in df.columns}


def preprocess_by_column_map(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Returns a dict of categorized series based on identified column roles.
    col_map: {column_name: clinical_category}
    """
    result = {}
    reverse = {}
    for col, category in col_map.items():
        if category not in reverse:
            reverse[category] = []
        reverse[category].append(col)

    for category, cols in reverse.items():
        if len(cols) == 1:
            result[category] = df[cols[0]]
        else:
            result[category] = df[cols]  # multiple columns with same category

    return result


def get_date_column(df: pd.DataFrame, col_map: dict) -> pd.Series | None:
    """Find and parse the date column if present."""
    for col, cat in col_map.items():
        if cat == "date" and col in df.columns:
            try:
                return pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                return None
    return None


def get_drug_column(df: pd.DataFrame, col_map: dict) -> pd.Series | None:
    """Return the drug name series if present."""
    for col, cat in col_map.items():
        if cat == "drug_name" and col in df.columns:
            return df[col].astype(str)
    return None


def build_summary(df: pd.DataFrame, col_map: dict) -> str:
    """Build a text summary of the dataset for LLM context."""
    lines = [
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns.",
        f"Columns and identified roles: {json.dumps(col_map)}",
        "",
        "Column statistics:",
    ]
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(
                    f"  {col} (numeric): min={df[col].min():.2f}, max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}, nulls={df[col].isna().sum()}"
                )
            else:
                top_vals = df[col].value_counts().head(5).to_dict()
                lines.append(
                    f"  {col} (categorical): {df[col].nunique()} unique values, "
                    f"top values={top_vals}, nulls={df[col].isna().sum()}"
                )
        except Exception:
            lines.append(f"  {col}: unable to summarize")

    return "\n".join(lines)