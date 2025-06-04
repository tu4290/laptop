import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional, List, Any
import dash_bootstrap_components as dbc
from dash import dcc, html
import logging # Added for instance_logger
import datetime # Added for _add_timestamp_annotation

# Assuming PLOTLY_TEMPLATE_DARK might be defined elsewhere or use a default
# For now, let's define a placeholder if it's used directly
DEFAULT_PLOTLY_TEMPLATE_DARK = "plotly_dark"

class MSPIVisualizerV2:
    def __init__(self, mspi_data: Optional[Dict[str, pd.DataFrame]] = None,
                 regime_model_results: Optional[Dict[str, Any]] = None, # Added regime_model_results
                 darkpool_analysis_df: Optional[pd.DataFrame] = None, # Added darkpool_analysis_df
                 underlying_price: Optional[float] = None,
                 plotly_template: str = DEFAULT_PLOTLY_TEMPLATE_DARK,
                 config: Optional[Dict[str, Any]] = None, # Added config
                 verbose: bool = False):
        self.mspi_data = mspi_data if mspi_data is not None else {}
        self.regime_model_results = regime_model_results if regime_model_results is not None else {}
        self.darkpool_analysis_df = darkpool_analysis_df
        self.underlying_price = underlying_price
        self.plotly_template = plotly_template
        self.config = config if config is not None else {} # Store config
        self.verbose = verbose

        # Basic logger
        self.instance_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if self.verbose:
            self.instance_logger.setLevel(logging.INFO)
        else:
            self.instance_logger.setLevel(logging.WARNING) # Or another appropriate level

        # Initialize commonly used styling attributes
        self.plot_bgcolor = 'rgba(34, 34, 34, 1)'
        self.paper_bgcolor = 'rgba(34, 34, 34, 1)'
        self.font_color = 'rgba(220, 220, 220, 1)'
        self.font_dict = dict(color=self.font_color) # Added font_dict
        self.grid_color = 'rgba(60, 60, 60, 1)'
        self.legend_bgcolor = 'rgba(34, 34, 34, 0.8)'
        self.axis_tick_font_size = 10
        self.axis_title_font_size = 12
        self.chart_height = 400
        self.ohlc_increasing_color = 'rgba(0, 204, 150, 1)' # Teal / Green
        self.ohlc_decreasing_color = 'rgba(255, 69, 0, 1)'   # OrangeRed / Red

        if self.verbose:
            print("MSPIVisualizerV2 initialized.")
            if self.mspi_data:
                print(f"MSPI Data Keys: {list(self.mspi_data.keys())}")
            if self.regime_model_results:
                print(f"Regime Model Results Keys: {list(self.regime_model_results.keys())}")
            if self.darkpool_analysis_df is not None and not self.darkpool_analysis_df.empty:
                print("Darkpool Analysis DataFrame loaded.")
            if self.underlying_price is not None:
                print(f"Underlying Price: {self.underlying_price}")

    def _create_chart_title_object(self, title_text: str, subtitle_text: Optional[str] = None) -> Dict:
        """Helper to create a Plotly title object with optional subtitle."""
        title = {
            'text': title_text,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': self.font_color}
        }
        if subtitle_text:
            title['text'] = f"{title_text}<br><sup_><span style='font-size: 12px;'>{subtitle_text}</span></sup>"
        return title

    # Placeholder for _get_config_value
    def _get_config_value(self, keys: List[str], default: Any = None) -> Any:
        """
        Retrieves a value from the instance's config dictionary using a list of keys.
        This is a simplified placeholder. A more robust version would handle
        nested dictionaries based on self.config.
        """
        if not hasattr(self, 'config') or not self.config:
            return default

        temp_config = self.config
        for key in keys:
            if isinstance(temp_config, dict) and key in temp_config:
                temp_config = temp_config[key]
            else:
                # self.instance_logger.debug(f"Config path {keys} not found, returning default: {default}")
                return default
        return temp_config

    # Placeholder for _add_timestamp_annotation
    def _add_timestamp_annotation(self, fig: go.Figure, timestamp: Optional[datetime.datetime] = None) -> go.Figure:
        """Adds a timestamp annotation to the figure. Placeholder."""
        if timestamp:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        fig.add_annotation(
            text=f"Updated: {timestamp_str}",
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.01,
            y=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=10, color=self.font_color)
        )
        return fig

    def create_darkpool_plausibility_chart(self,
                                           darkpool_analysis_df: Optional[pd.DataFrame],
                                           top_n: int = 15, # Configurable Top N
                                           current_price: Optional[float] = None) -> go.Figure:
        """
        Creates a bar chart of Darkpool Composite Plausibility Scores for top N strikes.
        Bar color can indicate Support/Resistance.
        """
        method_logger = self.instance_logger.getChild("create_darkpool_plausibility_chart")
        # Use self._get_config_value for accessing configuration

        # Default title from a more structured config access
        default_chart_title = f"Top Darkpool Levels by Plausibility"
        chart_title_from_config = self._get_config_value(
            ["visualization_settings", "darkpool_charts", "plausibility_chart_title"],
            default_chart_title
        )
        final_chart_title = chart_title_from_config.replace("{top_n}", str(top_n)) if "{top_n}" in chart_title_from_config else f"Top {top_n} Darkpool Levels by Plausibility"


        fig_height = self._get_config_value(["visualization_settings", "darkpool_charts", "plausibility_chart_height"], 600)
        plotly_template_to_use = self._get_config_value(["visualization_settings", "plotly_template"], "plotly_dark") # General template

        if darkpool_analysis_df is None or darkpool_analysis_df.empty:
            method_logger.info("No Darkpool analysis data provided. Returning empty chart.")
            # Pass title to _create_empty_figure, which might not exist, so ensure it can take it or use a default.
            # The existing _create_empty_figure takes 'message', not 'title'. Let's adapt.
            return self.create_empty_figure(message=f"{final_chart_title} - No Data")


        # Get column names from config, falling back to expected names
        # These should ideally align with DarkpoolReportColumns from elite_darkpool_analyzer
        strike_col = self._get_config_value(["darkpool_report_cols", "strike"], "strike")
        score_col = self._get_config_value(["darkpool_report_cols", "composite_plausibility_score"], "composite_plausibility_score")
        level_type_col = self._get_config_value(["darkpool_report_cols", "level_type"], "level_type")

        required_cols = [strike_col, score_col, level_type_col]
        missing_cols = [col for col in required_cols if col not in darkpool_analysis_df.columns]
        if missing_cols:
            method_logger.warning(f"Missing required columns for plausibility chart: {missing_cols}. Returning empty chart.")
            return self.create_empty_figure(message=f"{final_chart_title} - Missing Cols: {', '.join(missing_cols)}")


        df_to_plot = darkpool_analysis_df.copy()
        df_to_plot[score_col] = pd.to_numeric(df_to_plot[score_col], errors='coerce').fillna(0.0)
        df_to_plot[strike_col] = pd.to_numeric(df_to_plot[strike_col], errors='coerce') # Ensure strike is numeric for sorting/filtering
        df_to_plot = df_to_plot.dropna(subset=[strike_col])


        min_plausibility_display = self._get_config_value(["visualization_settings", "darkpool_charts", "min_plausibility_for_chart"], 0.01)
        df_to_plot = df_to_plot[df_to_plot[score_col] >= min_plausibility_display]

        if df_to_plot.empty:
            method_logger.info("No Darkpool levels met minimum plausibility for chart display.")
            return self.create_empty_figure(message=f"{final_chart_title} - No Significant Levels")

        df_to_plot = df_to_plot.sort_values(by=score_col, ascending=False).head(top_n)
        # Sort by strike for display after selecting top N by score
        df_to_plot = df_to_plot.sort_values(by=strike_col, ascending=True)

        default_colors = {
            "Support": "green", "Resistance": "red", "Contested": "yellow",
            "Potential Support": "lightgreen", "Potential Resistance": "lightcoral",
            "Uncertain": "grey", "Default": "blue"
        }
        color_map = self._get_config_value(["visualization_settings", "darkpool_charts", "level_type_colors"], default_colors)

        bar_colors = df_to_plot[level_type_col].map(lambda x: color_map.get(x, color_map.get("Default","blue")))

        fig = go.Figure(data=[
            go.Bar(
                x=df_to_plot[strike_col].astype(str),
                y=df_to_plot[score_col],
                text=df_to_plot[score_col].round(3), # Using 3 for plausibility score
                textposition='outside',
                marker_color=bar_colors,
                customdata=df_to_plot[[level_type_col, strike_col]].values, # Pass level_type and numeric strike for hover
                hovertemplate=(
                    f"<b>Strike: %{{customdata[1]:.2f}}</b><br>" # Numeric strike from customdata
                    f"Plausibility: %{{y:.3f}}<br>"
                    f"Level Type: %{{customdata[0]}}<extra></extra>"
                )
            )
        ])

        current_regime_str = str(self._get_config_value(['global_context', 'current_market_regime'], 'N/A'))

        fig.update_layout(
            title=self._create_chart_title_object(final_chart_title, subtitle=f"Regime: {current_regime_str}"),
            xaxis_title="Strike Price",
            yaxis_title="Composite Plausibility Score",
            template=self.plotly_template, # Uses template from class instance (loaded from config)
            paper_bgcolor=self.paper_bgcolor if hasattr(self, 'paper_bgcolor') else 'rgba(0,0,0,0)', # Use class attributes if they exist
            plot_bgcolor=self.plot_bgcolor if hasattr(self, 'plot_bgcolor') else 'rgba(0,0,0,0)',
            font=self.font_dict if hasattr(self, 'font_dict') else dict(color='white'), # Use class attributes
            height=fig_height,
            showlegend=False,
        )

        if current_price is not None and isinstance(current_price, (int, float)) and current_price > 0:
            fig.add_vline(
                x=current_price, line_width=1, line_dash="dash", line_color="cyan",
                annotation_text=f"Current: {current_price:.2f}", annotation_position="top right",
                annotation_font_size=10, annotation_font_color="cyan"
            )
            # Auto-adjust x-axis only if current_price is outside the range of plotted strikes
            # This part needs to handle strike_col being potentially non-numeric after .astype(str) for x-axis
            # So, use the numeric version of strikes for range calculation
            numeric_strikes_in_plot = pd.to_numeric(df_to_plot[strike_col], errors='coerce').dropna()
            if not numeric_strikes_in_plot.empty:
                min_strike_chart = numeric_strikes_in_plot.min()
                max_strike_chart = numeric_strikes_in_plot.max()
                current_price_num = float(current_price)

                if current_price_num < min_strike_chart or current_price_num > max_strike_chart:
                    buffer = (max_strike_chart - min_strike_chart) * 0.1 if max_strike_chart > min_strike_chart else 5
                    x_min_range = min(current_price_num, min_strike_chart) - buffer
                    x_max_range = max(current_price_num, max_strike_chart) + buffer
                    fig.update_xaxes(range=[x_min_range, x_max_range])

        fig = self._add_timestamp_annotation(fig, self._get_config_value(['global_context', 'current_fetch_timestamp'], None)) # Use global timestamp

        method_logger.info(f"Created Darkpool Plausibility chart with {len(df_to_plot)} levels.")
        # self._save_figure(fig, "DarkpoolPlausibility", symbol if symbol else "DP") # Assuming symbol is available or passed
        return fig

    def create_empty_figure(self, message: str = "No data available.", title: Optional[str] = None, height: Optional[int] = None) -> go.Figure:
        """Creates an empty figure with a message and optional title and height."""
        fig = go.Figure()

        display_message = message
        if title:
            display_message = f"<b>{title}</b><br>{message}"

        fig.add_annotation(
            text=display_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": self.font_color}
        )

        final_height = height if height is not None else self.chart_height

        layout_update = {
            'template': self.plotly_template,
            'plot_bgcolor': self.plot_bgcolor,
            'paper_bgcolor': self.paper_bgcolor,
            'xaxis': {'showgrid': False, 'zeroline': False, 'visible': False},
            'yaxis': {'showgrid': False, 'zeroline': False, 'visible': False},
            'height': final_height
        }
        if title: # Add title to layout if provided
            layout_update['title'] = self._create_chart_title_object(title)
            # Adjust annotation y if title is present to avoid overlap
            fig.layout.annotations[0].y = 0.45


        fig.update_layout(**layout_update)
        return fig

    # Placeholder for other methods that might exist in this class
    def some_other_chart_method(self):
        # This is just to illustrate that other methods would be here
        return self.create_empty_figure("Placeholder for another chart type")

if __name__ == '__main__':
    # Example Usage (basic test)
    visualizer = MSPIVisualizerV2(verbose=True)
    empty_fig = visualizer.create_empty_figure("Test Message for Empty Figure")
    # In a Dash app, you would typically show the figure:
    # empty_fig.show()
    print("MSPIVisualizerV2 created and empty figure generated.")

    # Example with Darkpool data (assuming structure)
    sample_darkpool_data = {
        'strike': [100, 105, 110, 115, 120],
        'composite_plausibility_score': [0.5, 0.8, 1.2, 0.9, 1.5],
        'level_type': ['Support', 'Resistance', 'Support', 'Contested', 'Resistance'],
        'gamma_concentration_mm': [10, 12, 8, 15, 9],
        'vanna_concentration_mm': [5, 6, 4, 7, 5]
    }
    dp_df = pd.DataFrame(sample_darkpool_data)

    visualizer_with_dp = MSPIVisualizerV2(darkpool_analysis_df=dp_df, underlying_price=112.5, verbose=True)
    # Later, a method like create_darkpool_plausibility_chart would be called here
    # e.g. dp_chart = visualizer_with_dp.create_darkpool_plausibility_chart()
    # dp_chart.show()
    print("MSPIVisualizerV2 created with sample Darkpool data.")
