import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations
from collections import Counter
import time


def run_pattern_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Pattern Agent: Discovers co-prescription patterns.
    Uses efficient pair counting (no mlxtend dependency required).
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "metrics": {}}

    drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
    patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in df.columns), None)

    if drug_col is None:
        result["status"] = "no_drug_col"
        result["summary"] = "No drug name column found. Cannot perform co-prescription pattern mining."
        return result

    try:
        # Top drugs overall
        drug_counts = df[drug_col].value_counts()
        top_10_drugs = drug_counts.head(10)
        fig_top = px.bar(
            x=top_10_drugs.values,
            y=top_10_drugs.index,
            orientation="h",
            title="Top 10 Most Prescribed Drugs",
            labels={"x": "Prescription Count", "y": "Drug"},
            color=top_10_drugs.values,
            color_continuous_scale="Teal",
            template="plotly_dark",
        )
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
            showlegend=False,
        )
        result["figures"].append(("Top Prescribed Drugs", fig_top))

        co_pairs = []
        n_patients = 0
        n_unique_pairs = 0
        avg_confidence = 0
        max_support = 0

        if patient_col:
            # Find drugs prescribed together per patient
            patient_drugs = df.groupby(patient_col)[drug_col].apply(list)
            n_patients = len(patient_drugs)
            pair_counts = Counter()
            for drugs in patient_drugs:
                unique_drugs = list(set(str(d) for d in drugs))
                if len(unique_drugs) > 1:
                    for pair in combinations(sorted(unique_drugs), 2):
                        pair_counts[pair] += 1

            n_unique_pairs = len(pair_counts)
            if pair_counts:
                top_pairs = pair_counts.most_common(15)
                pair_labels = [f"{a} + {b}" for (a, b), _ in top_pairs]
                pair_values = [count for _, count in top_pairs]

                # Support and Confidence for metrics
                max_support = top_pairs[0][1] / n_patients
                confidences = []
                for (a, b), count in top_pairs[:5]:
                    # Confidence A -> B = P(A & B) / P(A)
                    conf_a = count / drug_counts.get(a, 1)
                    conf_b = count / drug_counts.get(b, 1)
                    confidences.append(max(conf_a, conf_b))
                avg_confidence = np.mean(confidences) if confidences else 0

                fig_pairs = px.bar(
                    x=pair_values,
                    y=pair_labels,
                    orientation="h",
                    title="Top Co-Prescription Pairs",
                    labels={"x": "Co-occurrence Count", "y": "Drug Pair"},
                    color=pair_values,
                    color_continuous_scale="Oranges",
                    template="plotly_dark",
                )
                fig_pairs.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",
                    showlegend=False,
                )
                result["figures"].append(("Co-Prescription Pairs", fig_pairs))

                # Network/Sankey for top pairs
                top_10 = top_pairs[:10]
                sources, targets, values_list = [], [], []
                all_nodes = list(set([a for (a, b), _ in top_10] + [b for (a, b), _ in top_10]))
                node_idx = {n: i for i, n in enumerate(all_nodes)}

                for (a, b), count in top_10:
                    sources.append(node_idx[a])
                    targets.append(node_idx[b])
                    values_list.append(count)

                fig_sankey = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_nodes,
                        color=["#00C9A7"] * len(all_nodes),
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values_list,
                        color="rgba(0,201,167,0.3)",
                    ),
                ))
                fig_sankey.update_layout(
                    title="Drug Co-Prescription Flow (Sankey)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",
                )
                result["figures"].append(("Co-Prescription Sankey", fig_sankey))

                co_pairs = top_pairs
                result["summary"] = (
                    f"Pattern mining identified {len(pair_counts)} unique drug pairs across {len(patient_drugs)} patients.\n"
                    f"Top co-prescription: {top_pairs[0][0][0]} + {top_pairs[0][0][1]} ({top_pairs[0][1]} occurrences).\n"
                    f"Top 10 pairs shown in Sankey diagram and bar chart."
                )
            else:
                result["summary"] = "No co-prescription pairs found. Each patient appears to have only one drug."
        else:
            # No patient ID: just show drug frequency heatmap by other categorical cols
            cat_cols = [c for c, cat in col_map.items() if cat in ["region", "diagnosis", "prescriber", "gender"] and c in df.columns]
            if cat_cols:
                cat = cat_cols[0]
                top_drugs_list = top_10_drugs.index.tolist()
                cross = df[df[drug_col].isin(top_drugs_list)].groupby([drug_col, cat]).size().reset_index(name="Count")
                pivot = cross.pivot(index=drug_col, columns=cat, values="Count").fillna(0)
                fig_cross = px.imshow(
                    pivot,
                    title=f"Drug Prescriptions by {cat}",
                    color_continuous_scale="Teal",
                    template="plotly_dark",
                    aspect="auto",
                )
                fig_cross.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0")
                result["figures"].append((f"Drug × {cat} Heatmap", fig_cross))

            result["summary"] = (
                f"No patient ID column found. Showing drug frequency analysis.\n"
                f"Total unique drugs: {df[drug_col].nunique()}.\n"
                f"Most prescribed: {drug_counts.index[0]} ({drug_counts.iloc[0]} times)."
            )

        # Performance metrics
        duration = (time.perf_counter() - start_time) * 1000
        result["metrics"] = {
            "Confidence": f"{avg_confidence:.2f}",
            "Support": f"{max_support:.3f}",
            "Pairs": f"{n_unique_pairs}",
            "Patients": f"{n_patients}",
            "Execution": f"{duration:.1f}ms",
            "Model": "Apriori-lite"
        }

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Pattern agent error: {str(e)}"

    return result
