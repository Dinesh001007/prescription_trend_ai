import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def run_trend_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Trend Agent: Uses Prophet to forecast prescription trends over time.
    Falls back to rolling average trend if no date column found.
    """
    result = {"status": "ok", "figures": [], "summary": ""}

    date_col = next((c for c, cat in col_map.items() if cat == "date" and c in df.columns), None)
    drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
    qty_col = next((c for c, cat in col_map.items() if cat in ["quantity", "dosage", "frequency"] and c in df.columns), None)

    if date_col is None:
        result["status"] = "no_date"
        result["summary"] = "No date column identified. Cannot perform time-series trend analysis."
        _add_static_trend(df, col_map, result)
        return result

    try:
        df_trend = df.copy()
        df_trend["__date"] = pd.to_datetime(df_trend[date_col], errors="coerce")
        df_trend = df_trend.dropna(subset=["__date"])
        
        if qty_col:
            df_trend[qty_col] = pd.to_numeric(df_trend[qty_col], errors="coerce").fillna(0)

        if len(df_trend) < 5:
            result["status"] = "insufficient_data"
            result["summary"] = "Not enough dated records for trend analysis."
            return result

        # Aggregate: prescription count per time period
        df_trend = df_trend.sort_values("__date")
        date_range = (df_trend["__date"].max() - df_trend["__date"].min()).days

        if date_range < 30:
            freq = "D"
        elif date_range < 365:
            freq = "W"
        else:
            freq = "MS"

        if drug_col:
            # Per-drug trend
            top_drugs = df_trend[drug_col].value_counts().head(6).index.tolist()
            drug_trends = []
            for drug in top_drugs:
                sub = df_trend[df_trend[drug_col] == drug]
                if qty_col:
                    ts = sub.set_index("__date")[qty_col].resample(freq).sum()
                else:
                    ts = sub.set_index("__date").resample(freq).size()
                ts_df = ts.reset_index()
                ts_df.columns = ["date", "count"]
                ts_df["drug"] = drug
                drug_trends.append(ts_df)

            if drug_trends:
                all_trends = pd.concat(drug_trends)
                fig_multi = px.line(
                    all_trends,
                    x="date",
                    y="count",
                    color="drug",
                    title="Prescription Trend by Drug Over Time",
                    template="plotly_dark",
                    markers=True,
                )
                fig_multi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",
                    xaxis_title="Date",
                    yaxis_title="Prescription Volume",
                )
                result["figures"].append(("Drug Trends Over Time", fig_multi))

        # Overall trend + Prophet forecast
        if qty_col:
            overall_ts = df_trend.set_index("__date")[qty_col].resample(freq).sum()
        else:
            overall_ts = df_trend.set_index("__date").resample(freq).size()

        overall_ts = overall_ts.reset_index()
        overall_ts.columns = ["ds", "y"]
        overall_ts = overall_ts.dropna()

        if STATSMODELS_AVAILABLE and len(overall_ts) >= 10:
            try:
                periods = {"D": 30, "W": 12, "MS": 6}.get(freq, 12)
                seasonal_periods = {"D": 7, "W": 52, "MS": 12}.get(freq, 7)
                
                # Minimum data constraint for ETS
                if len(overall_ts) >= 2 * seasonal_periods:
                    model = ExponentialSmoothing(overall_ts["y"], trend="add", seasonal="add", seasonal_periods=seasonal_periods)
                else:
                    model = ExponentialSmoothing(overall_ts["y"], trend="add", seasonal=None)
                
                fit_model = model.fit(optimized=True)
                forecast_vals = fit_model.forecast(periods)
                
                # Build DS for forecast
                last_date = overall_ts["ds"].iloc[-1]
                if freq == "D":
                    future_ds = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
                elif freq == "W":
                    future_ds = [last_date + pd.Timedelta(weeks=i) for i in range(1, periods + 1)]
                else:
                    future_ds = [last_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]

                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(
                    x=overall_ts["ds"], y=overall_ts["y"],
                    mode="markers+lines", name="Actual",
                    line=dict(color="#00C9A7", width=2),
                    marker=dict(size=5),
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=future_ds, y=forecast_vals,
                    mode="lines", name="Forecast",
                    line=dict(color="#FFC300", width=2, dash="dash"),
                ))
                
                fig_forecast.update_layout(
                    title="Prescription Volume: Historical + Holt-Winters Forecast",
                    xaxis_title="Date",
                    yaxis_title="Prescription Volume",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#E8EAF0",
                )
                result["figures"].append(("Forecast", fig_forecast))

                last_actual = overall_ts["y"].iloc[-1]
                forecast_end = forecast_vals.iloc[-1]
                direction = "increasing" if forecast_end > last_actual else "decreasing"
                result["summary"] = (
                    f"Time-series trend analysis using Holt-Winters completed.\n"
                    f"Date range: {overall_ts['ds'].min().date()} to {overall_ts['ds'].max().date()}.\n"
                    f"Forecast horizon: {periods} periods ahead.\n"
                    f"Trend direction: {direction} (current: {last_actual:.0f} → forecast: {forecast_end:.0f})."
                )
            except Exception as e:
                result["summary"] = f"Forecast error: {str(e)}. Showing historical trend only."
                _plot_simple_trend(overall_ts, result)
        else:
            _plot_simple_trend(overall_ts, result)
            result["summary"] = (
                f"Trend analysis on {len(overall_ts)} time points.\n"
                f"Prophet unavailable or insufficient data; showing rolling average trend."
            )

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Trend agent error: {str(e)}"

    return result


def _plot_simple_trend(ts_df: pd.DataFrame, result: dict):
    """Fallback rolling average trend plot."""
    ts_df = ts_df.copy()
    ts_df["rolling"] = ts_df["y"].rolling(window=min(3, len(ts_df)), min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["ds"], y=ts_df["y"],
        mode="lines+markers", name="Actual",
        line=dict(color="#00C9A7"), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=ts_df["ds"], y=ts_df["rolling"],
        mode="lines", name="Rolling Avg",
        line=dict(color="#FFC300", dash="dash"),
    ))
    fig.update_layout(
        title="Prescription Volume Trend",
        xaxis_title="Date", yaxis_title="Volume",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8EAF0",
    )
    result["figures"].append(("Volume Trend", fig))


def _add_static_trend(df: pd.DataFrame, col_map: dict, result: dict):
    """Add frequency charts when no date column exists."""
    drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
    if drug_col:
        counts = df[drug_col].value_counts().head(15)
        fig = px.bar(
            x=counts.index, y=counts.values,
            title="Top Prescribed Drugs by Volume",
            labels={"x": "Drug", "y": "Count"},
            color=counts.values,
            color_continuous_scale="Teal",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
            showlegend=False,
        )
        result["figures"].append(("Top Drugs by Volume", fig))