# /home/ubuntu/dashboard_v2/layout.py
# -*- coding: utf-8 -*-
"""
Defines the layout structure for the Enhanced Options Dashboard V2.
Uses Dash Bootstrap Components for layout and includes placeholders for charts
and enhanced metric descriptions.
(Version 14 - Added Toggle for MSPI Chart Card)
"""

import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

# --- Define Fallback Functions FIRST ---
# These are used if the .utils or .styling modules cannot be imported.

def _fallback_get_config_value(keys: List[str], default: Any = None) -> Any:
    """Fallback function for get_config_value."""
    print(f"LAYOUT.PY WARNING: Using fallback for get_config_value for keys: {keys}")
    if keys == ["visualization_settings", "dashboard", "defaults", "symbol"]: return "/ES:XCME"
    if keys == ["visualization_settings", "dashboard", "defaults", "dte"]: return "0"
    if keys == ["visualization_settings", "dashboard", "defaults", "range_pct"]: return 5
    if keys == ["visualization_settings", "dashboard", "refresh_options"]:
        return [
            {"label": "Manual", "value": 0}, {"label": "30 Sec", "value": 30000},
            {"label": "1 Min", "value": 60000}, {"label": "5 Min", "value": 300000}
        ]
    if keys == ["visualization_settings", "dashboard", "defaults", "refresh_interval_ms"]: return 0
    if keys == ["visualization_settings", "dashboard", "range_slider_marks"]:
        return {str(i): f"{i}%" for i in [1, 2, 3, 5, 7, 10, 15, 20]}
    if keys == ["visualization_settings", "dashboard", "footer"]:
        return "© 2025 Elite Trading Systems Inc. (Fallback)"
    if keys == ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_options"]:
        return [ 
            {'label': 'Net Delta P (Heuristic)', 'value': 'heuristic_net_delta_pressure'},
            {'label': 'Net Gamma Flow', 'value': 'net_gamma_flow_at_strike'},
        ]
    if keys == ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_default_metric"]:
        return "heuristic_net_delta_pressure" 
    # Attempt to get darkpool_data for fallback if utils is not available
    if keys == ['darkpool_data']:
        print("LAYOUT.PY FALLBACK: Attempting to access darkpool_data directly (likely means APP_CONFIG not available via utils)")
        # This scenario is tricky as APP_CONFIG itself is usually populated via utils.
        # If utils.get_config_value is failing, this part of the fallback might not be very effective
        # unless APP_CONFIG was globally populated in enhanced_dashboard_v2.py and somehow accessible here.
        # For safety, return the default.
        return default
    return default

def _fallback_create_empty_figure(title: str = "Waiting for data...", height: Optional[int] = None, reason: Optional[str] = "N/A") -> go.Figure:
    """ Fallback function to create an empty figure if utils.create_empty_figure is unavailable. """
    print(f"LAYOUT.PY WARNING: Using fallback for create_empty_figure with title: {title}")
    fig_height = height if height is not None else 600
    plotly_template = "plotly_dark" # Default template for fallback
    # Check if PLOTLY_TEMPLATE_DARK is available globally from styling.py, otherwise use the local default
    if 'PLOTLY_TEMPLATE_DARK' in globals() and isinstance(globals()['PLOTLY_TEMPLATE_DARK'], dict) and 'layout' in globals()['PLOTLY_TEMPLATE_DARK']:
        plotly_template = globals()['PLOTLY_TEMPLATE_DARK']['layout'].get('template', plotly_template)

    fig = go.Figure()
    fig.update_layout(
        title={'text': f"<i>{title} (Utils Not Loaded)<br><small style='color:grey'>Reason: {reason}</small></i>", 'y':0.5, 'x':0.5, 'xanchor': 'center', 'yanchor': 'middle', 'font': {'color': 'grey', 'size': 16}},
        template=plotly_template, height=fig_height,
        xaxis={'visible': False, 'showgrid': False, 'zeroline': False},
        yaxis={'visible': False, 'showgrid': False, 'zeroline': False},
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Attempt Local Imports from .utils and .styling ---
PLOTLY_TEMPLATE_DARK = "plotly_dark" # Default value, will be overwritten if import is successful
try:
    from .utils import get_config_value, create_empty_figure
    from .styling import APP_THEME, PLOTLY_TEMPLATE_DARK as imported_plotly_template
    PLOTLY_TEMPLATE_DARK = imported_plotly_template # Update with imported template
    utils_styling_available = True
    print("LAYOUT.PY: Successfully imported from .utils and .styling.")
except ImportError as e:
    print(f"LAYOUT.PY WARNING: Failed to import from .utils or .styling: {e}. Using fallback functions/theme.")
    get_config_value = _fallback_get_config_value
    create_empty_figure = _fallback_create_empty_figure
    # PLOTLY_TEMPLATE_DARK remains "plotly_dark" (the string, not the dict object)
    utils_styling_available = False


# --- Component ID Constants ---
ID_SYMBOL_INPUT = "symbol-input"
ID_EXPIRATION_INPUT = "expiration-input"
ID_RANGE_SLIDER = "price-range-slider"
ID_INTERVAL_DROPDOWN = "interval-dropdown"
ID_FETCH_BUTTON = "fetch-button"
ID_STATUS_DISPLAY = "status-display"
ID_LOADING_SPINNER = "loading-status" 
ID_INTERVAL_TIMER = "interval-component"
ID_CACHE_STORE = "cache-key-store"
ID_CONFIG_STORE = "app-config-store" 

ID_NET_GREEK_FLOW_HEATMAP_CHART = "net-greek-flow-heatmap-chart"
ID_GREEK_FLOW_SELECTOR_IN_CARD = "greek-flow-selector-in-card" 

# --- NEW ID for MSPI Card Toggle ---
ID_MSPI_CHART_TOGGLE_SELECTOR = "mspi-chart-toggle-selector"

ID_MODE_TABS = "mode-tabs"
ID_MODE_CONTENT = "mode-content"
ID_TAB_MAIN_DASHBOARD = "tab-main-dashboard"
ID_TAB_SDAG_DIAGNOSTICS = "tab-sdag-diagnostics"
ID_TAB_DARKPOOL_ANALYSIS = "tab-darkpool-analysis"


# --- Chart Configuration: Blurbs ---
ENHANCED_BLURBS = {
    "mspi_heatmap": """<div class="metric-blurb"><h3>MSPI Heatmap / Alt View</h3>...</div>""", 
    "net_value_heatmap": """<div class="metric-blurb">...</div>""",
    "mspi_components": """<div class="metric-blurb">...</div>""",
    ID_NET_GREEK_FLOW_HEATMAP_CHART: """<div class="metric-blurb"><h3>Net Greek Flow & Pressure Heatmap</h3>...</div>""",
    "net_volval_comp": """<div class="metric-blurb">...</div>""",
    "combined_rolling_flow_chart": """<div class="metric-blurb">...</div>""",
    "volatility_regime": """<div class="metric-blurb">...</div>""",
    "time_decay": """<div class="metric-blurb">...</div>""",
    "sdag_multiplicative": """<div class="metric-blurb">...</div>""",
    "sdag_directional": """<div class="metric-blurb">...</div>""",
    "sdag_weighted": """<div class="metric-blurb">...</div>""",
    "sdag_volatility_focused": """<div class="metric-blurb">...</div>""",
    "key_levels": """<div class="metric-blurb">...</div>""",
    "trading_signals": """<div class="metric-blurb">...</div>""",
    "recommendations_table": """<div class="metric-blurb">...</div>"""
}

# --- Chart IDs for each mode ---
MAIN_DASHBOARD_CHART_IDS_ORDERED: List[str] = [
    "key_levels",                   
    "mspi_components",              
    "mspi_heatmap", # This ID will now be controlled by the new toggle
    ID_NET_GREEK_FLOW_HEATMAP_CHART, 
    "combined_rolling_flow_chart",  
    "volatility_regime",            
    "time_decay",                   
    "recommendations_table"         
]

SDAG_DIAGNOSTICS_CHART_IDS: List[str] = [
    "sdag_multiplicative", 
    "sdag_directional", 
    "sdag_weighted", 
    "sdag_volatility_focused"
]

ALL_CHART_IDS_FOR_FACTORY: List[str] = list(set(MAIN_DASHBOARD_CHART_IDS_ORDERED + SDAG_DIAGNOSTICS_CHART_IDS))


# --- Helper function to create a chart card ---
def create_chart_card(chart_id: str, blurb_html: str) -> dbc.Card:
    local_create_empty_figure = create_empty_figure 
    display_title_accordion = chart_id.replace('_', ' ').replace('-chart', '').title()
    
    card_body_children = []

    if chart_id == ID_NET_GREEK_FLOW_HEATMAP_CHART: 
        display_title_accordion = "Net Greek Flow & Pressure Heatmap"
        card_body_children.append(
            html.Div([
                dbc.Label("Select Metric:", html_for=ID_GREEK_FLOW_SELECTOR_IN_CARD, className="fw-bold control-label mb-1 small"),
                dcc.Dropdown(
                    id=ID_GREEK_FLOW_SELECTOR_IN_CARD,
                    options=get_config_value( 
                        ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_options"],
                        [{'label': 'Net Delta P (Heuristic)', 'value': 'heuristic_net_delta_pressure'}] 
                    ),
                    value=get_config_value( 
                        ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_default_metric"],
                        "heuristic_net_delta_pressure"
                    ),
                    clearable=False, className="mb-3 form-select-sm", searchable=False,
                    style={'fontSize': '0.85rem'} 
                )
            ], className="mb-2") 
        )
    elif chart_id == "mspi_heatmap": # This is the ID for the graph component in the MSPI card
        display_title_accordion = "MSPI View" # Generic title as content can change
        card_body_children.append(
            html.Div([
                dbc.Label("Select View:", html_for=ID_MSPI_CHART_TOGGLE_SELECTOR, className="fw-bold control-label mb-1 small"),
                dcc.Dropdown(
                    id=ID_MSPI_CHART_TOGGLE_SELECTOR,
                    options=[
                        {'label': 'MSPI Heatmap', 'value': 'mspi_heatmap'},
                        {'label': 'Net Volume Pressure (H) Heatmap', 'value': 'net_volume_pressure_heatmap'},
                        # Add more options here later if needed
                    ],
                    value='mspi_heatmap', # Default view
                    clearable=False, className="mb-3 form-select-sm", searchable=False,
                    style={'fontSize': '0.85rem'}
                )
            ], className="mb-2")
        )


    accordion_item = dbc.AccordionItem(
        dcc.Markdown(blurb_html, dangerously_allow_html=True, className="blurb-markdown-content"),
        title=f"About: {display_title_accordion}",
        item_id=f"blurb-accordion-item-{chart_id}",
    )
    accordion = dbc.Accordion(accordion_item, start_collapsed=True, flush=True, id=f"accordion-blurb-{chart_id}")
    
    graph_component_id = chart_id 
    
    card_body_children.append(
        dcc.Loading(
            id=f"loading-{chart_id}", type="circle", color="#007bff", fullscreen=False,
            children=[
                dcc.Graph(
                    id=graph_component_id,
                    figure=local_create_empty_figure(f"{display_title_accordion} - Loading..."),
                    config={'displayModeBar': True, 'scrollZoom': True, 'responsive': True},
                    className="chart-graph-obj"
                )
            ]
        )
    )
    
    card_content = [
        dbc.CardHeader(accordion, className="p-2 chart-card-header"), 
        dbc.CardBody(card_body_children, className="p-2 chart-card-body"),
    ]
    return dbc.Card(card_content, className="mb-4 shadow-sm chart-card-wrapper")

# --- Functions to get layout for each mode ---
def get_main_dashboard_mode_layout() -> html.Div:
    layout_children = []
    layout_children.append(dbc.Row([
        dbc.Col(create_chart_card("key_levels", ENHANCED_BLURBS.get("key_levels", "")), lg=6, md=12, className="chart-column"),
        dbc.Col(create_chart_card("mspi_components", ENHANCED_BLURBS.get("mspi_components", "")), lg=6, md=12, className="chart-column"),
    ], className="mb-3 chart-row"))
    
    layout_children.append(dbc.Row([
        dbc.Col(create_chart_card("mspi_heatmap", ENHANCED_BLURBS.get("mspi_heatmap", "")), lg=6, md=12, className="chart-column"),
        dbc.Col(create_chart_card(ID_NET_GREEK_FLOW_HEATMAP_CHART, ENHANCED_BLURBS.get(ID_NET_GREEK_FLOW_HEATMAP_CHART, "")), lg=6, md=12, className="chart-column"),
    ], className="mb-3 chart-row"))
    
    layout_children.append(dbc.Row([
        dbc.Col(create_chart_card("combined_rolling_flow_chart", ENHANCED_BLURBS.get("combined_rolling_flow_chart", "")), lg=12, md=12, className="chart-column"),
    ], className="mb-3 chart-row"))
    layout_children.append(dbc.Row([
        dbc.Col(create_chart_card("volatility_regime", ENHANCED_BLURBS.get("volatility_regime", "")), lg=6, md=12, className="chart-column"),
        dbc.Col(create_chart_card("time_decay", ENHANCED_BLURBS.get("time_decay", "")), lg=6, md=12, className="chart-column"),
    ], className="mb-3 chart-row"))
    layout_children.append(dbc.Row([
        dbc.Col(create_chart_card("recommendations_table", ENHANCED_BLURBS.get("recommendations_table", "")), lg=12, md=12, className="chart-column"),
    ], className="mb-3 chart-row"))
    return html.Div(layout_children)

def get_sdag_diagnostics_mode_layout() -> html.Div:
    chart_rows = []
    charts_in_current_row = []
    for i, chart_id in enumerate(SDAG_DIAGNOSTICS_CHART_IDS):
        blurb_html = ENHANCED_BLURBS.get(chart_id, f"<p>No blurb for {chart_id}.</p>")
        chart_col = dbc.Col(create_chart_card(chart_id, blurb_html), lg=6, md=12, className="chart-column")
        charts_in_current_row.append(chart_col)
        if len(charts_in_current_row) == 2 or i == len(SDAG_DIAGNOSTICS_CHART_IDS) - 1:
            chart_rows.append(dbc.Row(charts_in_current_row, className="mb-3 chart-row"))
            charts_in_current_row = []
    return html.Div(chart_rows)


def get_darkpool_mode_layout() -> html.Div:
    # local_get_config_value is already defined in layout.py (refers to the one imported or fallback)

    # Define expected columns for the table based on DarkpoolReportColumns
    # These names should align with what EliteDarkpoolAnalyzer.config.cols_report produces
    # Using common/expected names here; callback will map actual data.

    # For the table, we'll define columns in the callback based on the DataFrame.
    # Here, we just set up the DataTable component.

    darkpool_table_id = "darkpool-levels-table"
    darkpool_chart_id = "darkpool-plausibility-chart"

    # Default empty figure for the chart placeholder
    # The actual figure will be generated by a callback
    # Use the globally available create_empty_figure (either from .utils or fallback)
    initial_chart_figure = create_empty_figure(title="Darkpool Plausibility Chart - Loading...", height=600)

    return html.Div([
        html.H3("Elite Darkpool Analysis", className="mb-3"),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H4("Darkpool Plausibility Chart", className="card-title"),
                        dcc.Loading(
                            id=f"loading-{darkpool_chart_id}", type="circle",
                            children=[
                                dcc.Graph(
                                    id=darkpool_chart_id,
                                    figure=initial_chart_figure, # Initial empty figure
                                    config={'displayModeBar': True, 'scrollZoom': True}
                                )
                            ]
                        )
                    ]), className="mb-4 shadow-sm"
                ), md=12 # Make chart full width initially or md=7 if table is md=5
            ),
        ], className="mb-3"),

        dbc.Card(
            dbc.CardBody([
                html.H4("Identified Darkpool Levels", className="card-title"),
                dcc.Loading(
                    id=f"loading-{darkpool_table_id}", type="default", # Use default spinner for table
                    children=[
                        dash_table.DataTable(
                            id=darkpool_table_id,
                            # Columns will be set by callback based on incoming DataFrame
                            # Data will be set by callback
                            page_size=15,
                            style_as_list_view=True,
                            sort_action='native',
                            filter_action='native',
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'border': '1px solid rgb(50,50,50)'
                            },
                            style_cell={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white',
                                'border': '1px solid rgb(70, 70, 70)',
                                'textAlign': 'left',
                                'padding': '8px',
                                'minWidth': '100px', 'width': '150px', 'maxWidth': '250px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                            },
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(40, 40, 40)'},
                                # Conditional styling for S/R Level Type can be added in callback
                            ],
                            tooltip_data=[ # Enable tooltips for truncated cell data
                                {
                                    column: {'value': str(value), 'type': 'markdown'}
                                    for column, value in row.items()
                                } for row in [{}] # Placeholder, will be updated by callback
                            ],
                            tooltip_duration=None, # Keep tooltip open until mouse out
                             style_filter={
                                'backgroundColor': 'rgb(60,60,60)',
                                'color': 'white',
                                'border': '1px solid rgb(80,80,80)'
                            },
                        )
                    ]
                )
            ]), className="mb-4 shadow-sm"
        ),

        # Placeholder for Methodology Texts if still desired - remove if not part of this iteration
        # dbc.Row([
        #     dbc.Col(dbc.Card(dbc.CardBody([html.H4("Methodology Overview"), dcc.Markdown(id="darkpool-methodology-overview-text")]), className="metric-blurb shadow-sm"), md=6),
        #     dbc.Col(dbc.Card(dbc.CardBody([html.H4("Methodology Relationships"), dcc.Markdown(id="darkpool-methodology-relationships-text")]), className="metric-blurb shadow-sm"), md=6),
        # ], className="mb-3")

    ], className="darkpool-tab-container") # Add a class for potential tab-specific styling


# --- Main Layout Definition ---
def get_main_layout() -> dbc.Container:
    local_get_config_value = get_config_value 

    controls = dbc.Card(
        dbc.Row(
            [
                dbc.Col( 
                    [
                        html.Div([
                            dbc.Label("Symbol:", html_for=ID_SYMBOL_INPUT, className="fw-bold control-label"),
                            dbc.Input(id=ID_SYMBOL_INPUT, type="text", value=local_get_config_value(["visualization_settings", "dashboard", "defaults", "symbol"], "/ES:XCME"), placeholder="Enter symbol", className="mb-2 form-control-sm", debounce=True),
                        ]),
                        html.Div([
                            dbc.Label("DTE:", html_for=ID_EXPIRATION_INPUT, className="fw-bold control-label"),
                            dbc.Input(id=ID_EXPIRATION_INPUT, type="text", value=local_get_config_value(["visualization_settings", "dashboard", "defaults", "dte"], "0"), placeholder="e.g., 0, 0-7", className="mb-2 form-control-sm", debounce=True),
                        ]),
                    ], md=4, sm=12, className="control-column", 
                ),
                dbc.Col( 
                    [
                        dbc.Label("Strike Range %:", html_for=ID_RANGE_SLIDER, className="fw-bold control-label"),
                        dcc.Slider(id=ID_RANGE_SLIDER, min=1, max=20, step=1, value=local_get_config_value(["visualization_settings", "dashboard", "defaults", "range_pct"], 5), marks=local_get_config_value(["visualization_settings", "dashboard", "range_slider_marks"], {i: str(i) for i in range(1, 21, 2)}), tooltip={"placement": "bottom", "always_visible": False}, className="mb-2 pt-3 control-slider"),
                    ], md=4, sm=12, className="control-column", 
                ),
                dbc.Col( 
                    [
                         html.Div([
                            dbc.Label("Refresh:", html_for=ID_INTERVAL_DROPDOWN, className="fw-bold control-label"),
                            dbc.Select(id=ID_INTERVAL_DROPDOWN, options=local_get_config_value(["visualization_settings", "dashboard", "refresh_options"], [{'label': 'Manual', 'value': 0}]), value=local_get_config_value(["visualization_settings", "dashboard", "defaults", "refresh_interval_ms"], 0), className="mb-2 form-select-sm"),
                         ]),
                         html.Div([
                            dbc.Button("Fetch Data", id=ID_FETCH_BUTTON, color="primary", n_clicks=0, className="w-100 mt-2 btn-sm"),
                         ]),
                    ], md=4, sm=12, className="control-column align-self-end", 
                ),
            ], className="g-3 align-items-stretch", 
        ),
        body=True, className="mb-4 shadow-sm control-panel-card",
    )

    status_bar = dbc.Row(
        dbc.Col(html.Div(id=ID_STATUS_DISPLAY, className="p-2 text-center rounded status-bar-default-style", style={'backgroundColor': '#2c3e50', 'color': '#ecf0f1', 'minHeight': '40px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}), width=12),
        className="mb-3",
    )

    mode_tabs = dbc.Tabs(
        [
            dbc.Tab(label="Main Dashboard", tab_id=ID_TAB_MAIN_DASHBOARD, className="fw-bold", active_label_class_name="fw-bolder text-success"),
            dbc.Tab(label="SDAG Diagnostics", tab_id=ID_TAB_SDAG_DIAGNOSTICS, className="fw-bold", active_label_class_name="fw-bolder text-success"),
            dbc.Tab(label="Darkpool Analysis", tab_id=ID_TAB_DARKPOOL_ANALYSIS, className="fw-bold", active_label_class_name="fw-bolder text-success"),
        ],
        id=ID_MODE_TABS,
        active_tab=ID_TAB_MAIN_DASHBOARD, 
        className="mb-3 custom-tabs-container" 
    )

    main_app_layout = dbc.Container(
        [
            dcc.Store(id=ID_CACHE_STORE, storage_type='memory'),
            dcc.Store(id=ID_CONFIG_STORE, storage_type='memory', data=local_get_config_value([], {})), 
            dcc.Interval(
                id=ID_INTERVAL_TIMER,
                interval=int(local_get_config_value(["visualization_settings", "dashboard", "defaults", "refresh_interval_ms"], 0)),
                n_intervals=0,
                disabled=(int(local_get_config_value(["visualization_settings", "dashboard", "defaults", "refresh_interval_ms"], 0)) <= 0),
            ),
            controls,
            status_bar,
            mode_tabs, 
            html.Div(id=ID_MODE_CONTENT, className="mt-4 chart-layout-container"), 
        ],
        fluid=True,
        className="dbc main-app-container", 
    )
    return main_app_layout

CHART_IDS = ALL_CHART_IDS_FOR_FACTORY
