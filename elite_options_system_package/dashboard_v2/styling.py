# /home/ubuntu/dashboard_v2/styling.py

import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# --- Theme ---
# User requested a dark theme. We'll use a DBC dark theme and can customize further.
# For example, using dbc.themes.CYBORG or dbc.themes.DARKLY
# This will be set in the main app instance: external_stylesheets=[dbc.themes.CYBORG]
APP_THEME = dbc.themes.CYBORG # Placeholder, can be changed in main app script

# --- General Styles ---
STYLE_APP_WRAPPER = {
    "backgroundColor": "#1a1a1a", # Dark background
    "color": "#f0f0f0", # Light text
    "padding": "20px",
    "fontFamily": "'Segoe UI', Arial, sans-serif",
    "minHeight": "100vh"
}

STYLE_H1_TITLE = {
    "textAlign": "center",
    "color": "#e0e0e0",
    "marginBottom": "30px",
    "fontSize": "2.5em",
    "fontWeight": "bold",
    "textShadow": "2px 2px 4px #000000"
}

STYLE_FOOTER = {
    "marginTop": "50px",
    "paddingTop": "20px",
    "borderTop": "1px solid #444",
    "textAlign": "center",
    "fontSize": "0.9em",
    "color": "#777777"
}

# --- Control Panel Styles ---
STYLE_CONTROL_PANEL_WRAPPER = {
    "width": "95%",
    "maxWidth": "1200px",
    "margin": "0 auto 30px auto", # Centered
}

STYLE_CONTROL_BOX_CARD = { # To be used with dbc.Card
    "padding": "20px",
    "backgroundColor": "#282828",
    "borderRadius": "10px",
    "border": "1px solid #404040",
    "boxShadow": "0 6px 15px rgba(0,0,0,0.4)"
}

STYLE_CONTROL_ROW = { # For dbc.Row
    "marginBottom": "20px",
    "alignItems": "flex-end" # Align items at the bottom for mixed height controls
}

STYLE_LABEL = {
    "display": "block",
    "marginBottom": "8px",
    "fontWeight": "bold",
    "color": "#cccccc"
}

STYLE_INPUT = { # General style for dcc.Input, dcc.Dropdown etc. if not using DBC specific ones
    "width": "100%",
    "padding": "10px",
    "borderRadius": "5px",
    "border": "1px solid #555",
    "backgroundColor": "#3c3c3c",
    "color": "#eeeeee",
    "fontSize": "1em"
}

STYLE_BUTTON_PRIMARY = { # For main action buttons
    "width": "100%",
    "padding": "12px",
    "fontSize": "1.1em",
    "border": "none",
    "borderRadius": "5px",
    "cursor": "pointer",
    "transition": "background-color 0.2s ease, transform 0.1s ease",
    "boxShadow": "0 2px 5px rgba(0,0,0,0.2)"
}

STYLE_STATUS_DISPLAY = {
    "marginTop": "15px",
    "textAlign": "center",
    "minHeight": "25px",
    "padding": "8px",
    "borderRadius": "5px",
    "fontSize": "0.95em"
}

# --- Chart Styles ---
STYLE_CHART_ROW = { # For dbc.Row containing charts
    "marginBottom": "40px", # Increased from 25px for more vertical space between rows
}

STYLE_CHART_CARD = { # To be used with dbc.Card for each chart
    "backgroundColor": "#232323",
    "padding": "20px", # Increased from 15px for more internal breathing room
    "borderRadius": "8px",
    "border": "1px solid #383838",
    "height": "100%" # Ensure cards in a row have same height if using dbc.Row(align="stretch")
}

STYLE_CHART_TITLE_CONTAINER = {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "marginBottom": "10px" # Space between title and chart body
}

STYLE_CHART_TITLE = {
    "color": "#d0d0d0",
    "fontSize": "1.2em",
    "fontWeight": "bold",
}

STYLE_INFO_ICON = {
    "cursor": "pointer",
    "color": "#007bff",
    "fontSize": "1.1em"
}

STYLE_LOADING_WRAPPER = {
    "minHeight": "400px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "backgroundColor": "rgba(35,35,35,0.7)",
    "borderRadius": "8px"
}

# --- Plotly Figure Templates/Layouts ---
PLOTLY_TEMPLATE_DARK = {
    "layout": go.Layout(
        template="plotly_dark",
        font=dict(family="'Segoe UI', Arial, sans-serif", size=12, color="#f0f0f0"),
        title_font_size=18,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="#444444",
            linecolor="#555555",
            zerolinecolor="#666666",
            showgrid=True,
            gridwidth=1,
            title_font_color="#aaaaaa",
            tickfont_color="#aaaaaa"
        ),
        yaxis=dict(
            gridcolor="#444444",
            linecolor="#555555",
            zerolinecolor="#666666",
            showgrid=True,
            gridwidth=1,
            title_font_color="#aaaaaa",
            tickfont_color="#aaaaaa"
        ),
        legend=dict(
            bgcolor="rgba(40,40,40,0.8)",
            bordercolor="#555555",
            borderwidth=1,
            font_color="#f0f0f0"
        ),
    )
}

# --- Specific Component Styles (Example) ---
STYLE_TOOLTIP = {
    "position": "absolute",
    "padding": "8px",
    "backgroundColor": "#111111",
    "border": "1px solid #007bff",
    "borderRadius": "4px",
    "color": "#f0f0f0",
    "fontSize": "0.9em",
    "boxShadow": "0px 0px 10px rgba(0,123,255,0.5)",
    "pointerEvents": "none",
    "zIndex": "1000"
}
