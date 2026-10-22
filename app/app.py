import base64
import io
import logging

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 20

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server
app.title = "Tablero DIE - Análisis de Legajos"


# ---------- Layout ----------
app.layout = dbc.Container([
    dbc.Row(dbc.Col([
        html.H1("📊 Tablero DIE", className="text-center mt-4 mb-2"),
        html.P(
            "Subí el CSV de legajos y obtené el reporte de seguimiento",
            className="text-center text-muted mb-4",
        ),
    ])),
    dbc.Row(dbc.Col([
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("📂", style={"fontSize": "56px", "lineHeight": "1"}),
                html.P(
                    "Arrastrá o hacé clic para seleccionar el CSV",
                    style={"margin": "8px 0 0 0", "fontSize": "15px", "color": "#555"},
                ),
            ], style={
                "display": "flex", "flexDirection": "column",
                "alignItems": "center", "justifyContent": "center",
                "height": "150px",
            }),
            style={
                "width": "100%",
                "border": "2px dashed #ced4da",
                "borderRadius": "12px",
                "backgroundColor": "#f8f9fa",
                "cursor": "pointer",
                "padding": "10px",
                "boxSizing": "border-box",
            },
            multiple=False,
            accept=".csv",
            max_size=MAX_FILE_SIZE_MB * 1024 * 1024,
        ),
        html.Div(id="upload-error", className="mt-2"),
    ], width=12)),
    html.Div(id="report-container", className="mt-4"),
], fluid=True)


# ---------- Helpers ----------
def _read_csv_robusto(decoded: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(
                io.StringIO(decoded.decode(encoding)),
                dtype=str,
                keep_default_na=False,
                na_values=[""],
                sep=None,
                engine="python",
            )
            
            new_cols = []
            seen = {}
            for i, c in enumerate(df.columns):
                # Forzar string y limpiar espacios
                name = str(c).strip() if c is not None else ""
                # Si quedó vacío o es tipo "Unnamed: X", generar nombre
                if not name or name.lower().startswith("unnamed"):
                    name = f"col_{i}"
                # Resolver duplicados
                if name in seen:
                    seen[name] += 1
                    name = f"{name}_{seen[name]}"
                else:
                    seen[name] = 0
                new_cols.append(name)
            df.columns = new_cols

            logger.info("CSV leído con encoding=%s shape=%s", encoding, df.shape)

            return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_error = e
            continue
    raise ValueError(f"No se pudo leer el CSV. Último error: {last_error}")


def _parse_upload(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    return _read_csv_robusto(decoded)


def _tabla_nulos(df: pd.DataFrame) -> pd.DataFrame:
    t = df.isna().sum().to_frame("nulos")
    t["%"] = (t["nulos"] / len(df) * 100).round(2)
    return t.sort_values("%", ascending=False).reset_index().rename(columns={"index": "Columna"})


def _tabla_unicos(df: pd.DataFrame, col: str, top: int = 15) -> pd.DataFrame:
    vc = df[col].value_counts(dropna=False).head(top)
    vc = vc.reset_index()
    vc.columns = [col, "Cantidad"]
    return vc


def _resumen(df: pd.DataFrame) -> dict:
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "dup_codigo": int(df["Codigo"].duplicated().sum()) if "Codigo" in df.columns else None,
        "dup_dni": int(df["DNI"].duplicated().sum()) if "DNI" in df.columns else None,
        "dup_total": int(df.duplicated().sum()),
        "nulos_total": int(df.isna().sum().sum()),
    }


def _fig_estado(df: pd.DataFrame):
    if "Estado" not in df.columns:
        return None
    vc = df["Estado"].value_counts().reset_index()
    vc.columns = ["Estado", "Cantidad"]
    fig = px.bar(
        vc, x="Cantidad", y="Estado", orientation="h",
        title="Distribución por Estado",
        color="Cantidad", color_continuous_scale="Blues", text="Cantidad",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380, template="plotly_white")
    return fig


def _fig_motivo(df: pd.DataFrame):
    if "Motivo de último Rechazo Marian" not in df.columns:
        return None
    vc = df["Motivo de último Rechazo Marian"].value_counts().head(15).reset_index()
    vc.columns = ["Motivo", "Cantidad"]
    fig = px.bar(
        vc, x="Cantidad", y="Motivo", orientation="h",
        title="Top motivos de último rechazo",
        color="Cantidad", color_continuous_scale="Reds", text="Cantidad",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, template="plotly_white")
    return fig


def _fig_observaciones(df: pd.DataFrame):
    if "OBSERVACIONES REGULARIDAD" not in df.columns:
        return None
    vc = df["OBSERVACIONES REGULARIDAD"].value_counts().head(15).reset_index()
    vc.columns = ["Observación", "Cantidad"]
    fig = px.bar(
        vc, x="Cantidad", y="Observación", orientation="h",
        title="Top OBSERVACIONES REGULARIDAD",
        color="Cantidad", color_continuous_scale="Greens", text="Cantidad",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, template="plotly_white")
    return fig


def _cruce(df: pd.DataFrame, a: str, b: str):
    if a not in df.columns or b not in df.columns:
        return None
    ct = pd.crosstab(df[a], df[b], dropna=False)
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    ct = ct[ct.sum(axis=0).sort_values(ascending=False).index]
    return ct.reset_index()


def _tabla_dash(df: pd.DataFrame, max_rows: int = 30) -> dash_table.DataTable:
    df = df.copy()
    # Asegurar que todas las columnas tengan name/id string válido
    safe_cols = []
    for i, c in enumerate(df.columns):
        c_str = str(c) if c is not None else ""
        if not c_str or c_str.lower().startswith("unnamed"):
            c_str = f"col_{i}"
        safe_cols.append(c_str)
    df.columns = safe_cols
    
    return dash_table.DataTable(
        data=df.head(max_rows).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left", "padding": "6px", "fontSize": "13px",
            "maxWidth": "280px", "overflow": "hidden", "textOverflow": "ellipsis",
        },
        style_header={"backgroundColor": "#2563eb", "color": "white", "fontWeight": "bold"},
    )


def _render_reporte(df: pd.DataFrame, filename: str):
    res = _resumen(df)
    children = [
        dbc.Alert(
            [html.Strong(f"✅ {filename}"), html.Br(),
             f"{res['filas']:,} filas · {res['columnas']} columnas"],
            color="success", className="mb-3",
        ),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Duplicados por Código", className="text-muted mb-1"),
                html.H4(res["dup_codigo"] if res["dup_codigo"] is not None else "—"),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Duplicados por DNI", className="text-muted mb-1"),
                html.H4(res["dup_dni"] if res["dup_dni"] is not None else "—"),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Filas totalmente duplicadas", className="text-muted mb-1"),
                html.H4(res["dup_total"]),
            ])), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Celdas nulas totales", className="text-muted mb-1"),
                html.H4(f"{res['nulos_total']:,}"),
            ])), width=3),
        ], className="mb-4"),
    ]

    children.append(html.H5("🧩 Nulos por columna", className="mt-4"))
    children.append(_tabla_dash(_tabla_nulos(df), max_rows=50))

    categoricas = [
        "Estado",
        "Documento",
        "OBSERVACIONES REGULARIDAD",
        "OBSERVACIÓN 01-10",
        "Instancia de último Rechazo Marian",
        "Motivo de último Rechazo Marian",
    ]
    children.append(html.H5("🏷️ Valores más frecuentes por columna categórica", className="mt-4"))
    for col in categoricas:
        if col in df.columns:
            children.append(html.H6(col, className="mt-3"))
            children.append(_tabla_dash(_tabla_unicos(df, col), max_rows=20))

    for fig_fn, titulo in [
        (_fig_estado, "Estado"),
        (_fig_motivo, "Motivo de último rechazo"),
        (_fig_observaciones, "Observaciones de regularidad"),
    ]:
        fig = fig_fn(df)
        if fig is not None:
            children.append(html.H5(f"📈 {titulo}", className="mt-4"))
            children.append(dcc.Graph(
            figure=fig,
            config={"displayModeBar": False},
            style={"height": "420px", "width": "100%"},
        ))

    children.append(html.H5("🔀 Cruces", className="mt-4"))
    cruces = [
        ("Estado", "Motivo de último Rechazo Marian", "Estado × Motivo de rechazo"),
        ("Estado", "Instancia de último Rechazo Marian", "Estado × Instancia de rechazo"),
        ("Estado", "OBSERVACIONES REGULARIDAD", "Estado × Observaciones"),
    ]
    print("=== COLUMNAS REALES ===")
    for c in df.columns:
        print(repr(c))
    print("=======================")
    for a, b, titulo in cruces:
        ct = _cruce(df, a, b)
        if ct is not None:
            children.append(html.H6(titulo, className="mt-3"))
            children.append(_tabla_dash(ct, max_rows=30))

    if "Estado" in df.columns and "Motivo de último Rechazo Marian" in df.columns:
        baja = df["Estado"].str.contains("ART. 16", na=False)
        sin_motivo = df["Motivo de último Rechazo Marian"].isna()
        children.append(html.H5("⚠️ Consistencia: bajas ART.16", className="mt-4"))
        children.append(dbc.Alert(
            f"Bajas ART.16: {int(baja.sum())} · "
            f"Bajas ART.16 sin motivo de rechazo: {int((baja & sin_motivo).sum())}",
            color="warning",
        ))

    return html.Div(children)


# ---------- Callback ----------
@app.callback(
    Output("report-container", "children"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        return html.Div()
    try:
        df = _parse_upload(contents, filename)
    except Exception as e:
        logger.exception("Error procesando CSV")
        return dbc.Alert(
            [html.Strong("❌ Error al procesar el archivo"), html.Br(), str(e)],
            color="danger",
        )
    return _render_reporte(df, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)