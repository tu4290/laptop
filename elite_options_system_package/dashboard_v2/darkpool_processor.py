# /home/ubuntu/dashboard_v2/darkpool_processor.py
# -*- coding: utf-8 -*-
import pandas as pd
import re
from typing import Dict, Optional, Tuple, List

def parse_darkpool_report(markdown_file_path: str) -> Optional[Dict[str, any]]:
    """
    Parses the Darkpool Analysis Report markdown file to extract:
    - Ultra Darkpool Levels table into a pandas DataFrame.
    - Methodology Overview text.
    - Methodology Relationships text.

    Args:
        markdown_file_path (str): The path to the markdown file.

    Returns:
        Optional[Dict[str, any]]: A dictionary containing:
            'ultra_levels_df': pd.DataFrame,
            'methodology_overview_text': str,
            'methodology_relationships_text': str
        Returns None if critical sections are not found or parsing fails.
    """
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {markdown_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {markdown_file_path}: {e}")
        return None

    results = {}

    # --- Extract Ultra Darkpool Levels table ---
    ultra_levels_df = None
    try:
        # More flexible regex for the start of the table section
        # It looks for "## Ultra Darkpool Levels" followed by optional non-header lines, then the table header
        table_start_match = re.search(r"## Ultra Darkpool Levels\s*^(.*?)^(\| Strike.*\|)", content, re.MULTILINE | re.DOTALL | re.IGNORECASE)

        if table_start_match:
            # The actual table content starts after the "## Ultra Darkpool Levels" header
            # and continues until "## Methodology Relationships"
            table_block_match = re.search(r"## Ultra Darkpool Levels\s*\n(.*?)(?=## Methodology Overview|## Methodology Relationships|## Conclusion|$)", content, re.DOTALL | re.IGNORECASE)

            if table_block_match:
                table_text_content = table_block_match.group(1).strip()
                lines = table_text_content.split('\n')

                header_line_index = -1
                separator_line_index = -1

                # Find header and separator lines
                for i, line in enumerate(lines):
                    stripped_line = line.strip()
                    if stripped_line.startswith("| Strike") and stripped_line.endswith("|"):
                        header_line_index = i
                    elif header_line_index != -1 and i == header_line_index + 1 and \
                         stripped_line.startswith("|--") and stripped_line.endswith("|") and \
                         all(c in '-| ' for c in stripped_line): # Ensure it's a proper separator
                        separator_line_index = i
                        break # Found header and separator

                if header_line_index != -1 and separator_line_index != -1:
                    header_line = lines[header_line_index].strip()
                    columns = [col.strip() for col in header_line.strip('|').split('|')]

                    data_lines = []
                    for i in range(separator_line_index + 1, len(lines)):
                        line_content = lines[i].strip()
                        if line_content.startswith("|") and line_content.endswith("|"):
                            data_lines.append(line_content)
                        elif not line_content: # Stop if an empty line is encountered after table rows
                            break
                        # else: could be non-table text, ignore for now unless stricter parsing is needed

                    parsed_data = []
                    if data_lines:
                        for line in data_lines:
                            values = [val.strip() for val in line.strip('|').split('|')]
                            if len(values) == len(columns):
                                parsed_data.append(values)
                            else:
                                print(f"Warning: Row value count mismatch. Expected {len(columns)}, got {len(values)}. Row: '{line}'")

                        if parsed_data:
                            ultra_levels_df = pd.DataFrame(parsed_data, columns=columns)
                            for col in ["Plausibility", "Gamma Concentration", "Delta Exposure", "Flow (15m)", "Charm Effect", "Strike"]: # Added Strike for safety
                                if col in ultra_levels_df.columns:
                                    # Attempt to clean non-numeric characters before conversion, e.g., '$', '%'
                                    if ultra_levels_df[col].dtype == 'object': # Only if it's string-like
                                        ultra_levels_df[col] = ultra_levels_df[col].str.replace(r'[^\d\.\-]', '', regex=True)
                                    ultra_levels_df[col] = pd.to_numeric(ultra_levels_df[col], errors='coerce')
                            results['ultra_levels_df'] = ultra_levels_df
                        else:
                            print("Warning: Ultra Darkpool Levels table found but no data rows parsed from data_lines.")
                    else:
                        print("Warning: Ultra Darkpool Levels table header and separator found, but no data lines followed.")
                else:
                    print("Warning: Ultra Darkpool Levels table header or separator line not correctly identified in the block.")
            else:
                print("Warning: Could not isolate Ultra Darkpool Levels table text block under its header.")
        else:
            print("Warning: 'Ultra Darkpool Levels' section header followed by a table-like structure not found.")
    except Exception as e:
        print(f"Error parsing Ultra Darkpool Levels table: {e}")
        import traceback
        traceback.print_exc()


    # --- Extract Methodology Overview text ---
    try:
        # Look for "Methodology Overview" then capture until the next "## " or end of string
        overview_match = re.search(r"## Methodology Overview\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL | re.IGNORECASE)
        if overview_match:
            results['methodology_overview_text'] = overview_match.group(1).strip()
        else:
            print("Warning: 'Methodology Overview' section not found.")
            results['methodology_overview_text'] = "Methodology Overview section not found in the report."
    except Exception as e:
        print(f"Error extracting Methodology Overview: {e}")
        results['methodology_overview_text'] = "Error extracting Methodology Overview."

    # --- Extract Methodology Relationships text ---
    try:
        # Look for "Methodology Relationships" then capture until the next "## " or end of string
        relationships_match = re.search(r"## Methodology Relationships\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL | re.IGNORECASE)
        if relationships_match:
            results['methodology_relationships_text'] = relationships_match.group(1).strip()
        else:
            print("Warning: 'Methodology Relationships' section not found.")
            results['methodology_relationships_text'] = "Methodology Relationships section not found in the report."
    except Exception as e:
        print(f"Error extracting Methodology Relationships: {e}")
        results['methodology_relationships_text'] = "Error extracting Methodology Relationships."

    # Check if the critical DataFrame was parsed
    if 'ultra_levels_df' not in results or results['ultra_levels_df'] is None:
        print("Critical error: Ultra Darkpool Levels DataFrame could not be parsed. Returning None.")
        return None

    # If DataFrame is present, return results, even if text sections might have defaults
    return results


if __name__ == '__main__':
    # This is for testing the parser directly
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is in elite_options_system_package/dashboard_v2/
    # and the 'darkpool' directory is at the root of the repository,
    # which is three levels up from script_dir.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    # Default path for a test file, assuming 'darkpool' folder is at repo root
    default_md_path = os.path.join(repo_root, "darkpool", "Darkpool Analysis Report for SPY.md")

    # Allow overriding path via environment variable for easier testing in different contexts
    md_file_to_test = os.environ.get("DARKPOOL_TEST_FILE", default_md_path)

    print(f"Attempting to parse: {md_file_to_test}")

    if not os.path.exists(md_file_to_test):
        print(f"ERROR: Test file not found at {md_file_to_test}")
        print("Please ensure the file exists or set the DARKPOOL_TEST_FILE environment variable.")
    else:
        parsed_output = parse_darkpool_report(md_file_to_test)

        if parsed_output:
            print("\n--- Ultra Darkpool Levels DataFrame ---")
            if 'ultra_levels_df' in parsed_output and parsed_output['ultra_levels_df'] is not None:
                print(parsed_output['ultra_levels_df'].to_string()) # Use to_string for better console output
            else:
                print("DataFrame not found or empty.")

            print("\n--- Methodology Overview ---")
            print(parsed_output.get('methodology_overview_text', 'Not found.'))

            print("\n--- Methodology Relationships ---")
            print(parsed_output.get('methodology_relationships_text', 'Not found.'))
        else:
            print("\nFailed to parse the Darkpool report, or critical sections were missing.")
