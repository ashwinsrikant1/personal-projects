import streamlit as st
import anthropic
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import altair as alt

# Load environment variables from .env file
load_dotenv()

# Hardcoded API key placeholder (fallback)
ANTHROPIC_API_KEY = "# TODO: Add your Anthropic API key here"

# Daily goals for each profile
DAILY_GOALS = {
    "Ashwin": {
        "calories": 2200,
        "protein": 120,
        "fat": 70,
        "carbs": 210,
        "fiber": 35,
        "sugar": 30
    },
    "Nandhitha": {
        "calories": 1700,
        "protein": 80,
        "fat": 50,
        "carbs": 160,
        "fiber": 25,
        "sugar": 25
    }
}

def analyze_nutrition(food_description: str) -> dict:
    """
    Analyze food description using Claude API and return nutritional information.

    Args:
        food_description: Text description of food consumed

    Returns:
        Dictionary containing nutritional information (calories, protein, carbs, fat, sugar, fiber)
    """
    # Try multiple sources for API key:
    # 1. Streamlit secrets (for Streamlit Cloud deployment)
    # 2. Environment variable from .env (for local development)
    # 3. Hardcoded placeholder (fallback)
    api_key = None

    # Try Streamlit secrets first
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable
        api_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

    if not api_key or api_key.startswith("# TODO"):
        raise ValueError("Please set your Anthropic API key in Streamlit secrets or .env file")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze the following food description and provide detailed nutritional information.
Return ONLY a JSON object with the following fields (all numeric values):
- calories (kcal)
- protein (g)
- carbs (g)
- fat (g)
- sugar (g) - IMPORTANT: This should be ADDED SUGAR only, NOT total sugar. Do not include natural sugars from fruits, milk, etc.
- fiber (g)

Food description: {food_description}

Return only the JSON object, no other text."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text.strip()

        # Try to extract JSON from markdown code blocks or other formatting
        if "```json" in response_text:
            # Extract JSON from markdown code block
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            # Extract from generic code block
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "{" in response_text:
            # Try to extract just the JSON object
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            response_text = response_text[start:end]

        nutrition_data = json.loads(response_text)

        # Validate required fields
        required_fields = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']
        for field in required_fields:
            if field not in nutrition_data:
                raise ValueError(f"Missing required field: {field}")

        return nutrition_data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse nutrition data. Response was: {response_text[:200]}... Error: {e}")
    except ValueError:
        # Re-raise ValueError (for missing fields)
        raise
    except Exception as e:
        raise Exception(f"Error analyzing nutrition: {e}")


def save_to_csv(data: dict, filename: str = "nutrition_data.csv"):
    """
    Save nutrition data to CSV file.

    Args:
        data: Dictionary containing profile, date, food_description, and nutrition info
        filename: Name of CSV file to save to
    """
    # Create DataFrame from single record
    df_new = pd.DataFrame([data])

    # Check if file exists
    if os.path.exists(filename):
        # Load existing data and append
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filename, index=False)
    else:
        # Create new file
        df_new.to_csv(filename, index=False)


def load_from_csv(filename: str = "nutrition_data.csv") -> pd.DataFrame:
    """
    Load nutrition data from CSV file.

    Args:
        filename: Name of CSV file to load from

    Returns:
        DataFrame containing nutrition data
    """
    if os.path.exists(filename):
        # Read with explicit dtypes to avoid dtype warnings
        dtypes = {
            'profile': str,
            'date': str,
            'food_description': str,
            'calories': float,
            'protein': float,
            'carbs': float,
            'fat': float,
            'sugar': float,
            'fiber': float
        }
        return pd.read_csv(filename, dtype=dtypes)
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'profile', 'date', 'food_description',
            'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'
        ])


def update_entry(index: int, updated_data: dict, filename: str = "nutrition_data.csv"):
    """
    Update a specific entry in the CSV file.

    Args:
        index: DataFrame index of the entry to update
        updated_data: Dictionary with updated values
        filename: Name of CSV file
    """
    df = load_from_csv(filename)
    if not df.empty and index in df.index:
        for key, value in updated_data.items():
            # Ensure numeric columns are proper float type
            if key in ['calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']:
                df.at[index, key] = float(value)
            else:
                df.at[index, key] = value
        df.to_csv(filename, index=False)


def delete_entry(index: int, filename: str = "nutrition_data.csv"):
    """
    Delete a specific entry from the CSV file.

    Args:
        index: DataFrame index of the entry to delete
        filename: Name of CSV file
    """
    df = load_from_csv(filename)
    if not df.empty:
        df = df.drop(index)
        df.to_csv(filename, index=False)


def clear_all_data(profile: str, filename: str = "nutrition_data.csv"):
    """
    Clear all data for a specific profile.

    Args:
        profile: Profile name to clear data for
        filename: Name of CSV file
    """
    df = load_from_csv(filename)
    if not df.empty:
        df = df[df['profile'] != profile]
        if df.empty:
            # If no data left, delete the file
            import os
            if os.path.exists(filename):
                os.remove(filename)
        else:
            df.to_csv(filename, index=False)


def calculate_daily_summary(df: pd.DataFrame, profile: str, date: str) -> dict:
    """
    Calculate daily summary metrics for a specific profile and date.

    Args:
        df: DataFrame containing nutrition data
        profile: Profile name to filter by
        date: Date to filter by (YYYY-MM-DD format)

    Returns:
        Dictionary with summed nutritional values
    """
    filtered = df[(df['profile'] == profile) & (df['date'] == date)]

    if filtered.empty:
        return {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'sugar': 0,
            'fiber': 0,
            'entries': 0
        }

    return {
        'calories': filtered['calories'].sum(),
        'protein': filtered['protein'].sum(),
        'carbs': filtered['carbs'].sum(),
        'fat': filtered['fat'].sum(),
        'sugar': filtered['sugar'].sum(),
        'fiber': filtered['fiber'].sum(),
        'entries': len(filtered)
    }


def get_7day_trend(df: pd.DataFrame, profile: str, end_date: str = None) -> pd.DataFrame:
    """
    Get 7-day trend data for a specific profile.

    Args:
        df: DataFrame containing nutrition data
        profile: Profile name to filter by
        end_date: End date for the 7-day window (defaults to today)

    Returns:
        DataFrame with daily aggregated nutrition data for the last 7 days
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    end = datetime.strptime(end_date, "%Y-%m-%d")
    start = end - timedelta(days=6)  # 7 days including end date

    # Generate all dates in range
    date_range = pd.date_range(start=start, end=end, freq='D')
    dates_str = [d.strftime("%Y-%m-%d") for d in date_range]

    # Filter data for profile and date range
    profile_data = df[df['profile'] == profile].copy()

    # Group by date and sum macros
    if not profile_data.empty:
        daily_sums = profile_data.groupby('date').agg({
            'calories': 'sum',
            'protein': 'sum',
            'carbs': 'sum',
            'fat': 'sum',
            'sugar': 'sum',
            'fiber': 'sum'
        }).reset_index()
    else:
        daily_sums = pd.DataFrame(columns=['date', 'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'])

    # Create a DataFrame with all 7 dates, filling missing dates with 0
    trend_data = pd.DataFrame({'date': dates_str})
    trend_data = trend_data.merge(daily_sums, on='date', how='left').fillna(0)

    return trend_data


def main():
    st.set_page_config(page_title="Nutrition Tracker", layout="wide")

    st.title("Nutrition Tracker")
    st.markdown("Track your daily nutrition using AI-powered food analysis")

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")

        # Profile selection
        profile = st.selectbox(
            "Select Profile",
            options=["Ashwin", "Nandhitha"],
            index=0
        )

        # Date picker
        selected_date = st.date_input(
            "Date",
            value=datetime.now()
        )
        date_str = selected_date.strftime("%Y-%m-%d")

        # Food description text area
        food_description = st.text_area(
            "Food Description",
            placeholder="E.g., 2 scrambled eggs, 1 slice of whole wheat toast with butter, 1 cup of coffee with milk",
            height=150
        )

        # Split meal option
        st.markdown("---")
        split_meal = st.checkbox("Split this meal between Ashwin and Nandhitha")

        split_percentage = 50  # Default 50/50 split
        if split_meal:
            st.info("🍽️ Splitting meal: Each person will get their portion of the macros")
            split_percentage = st.slider(
                f"{profile}'s portion",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                help=f"Adjust how much {profile} ate vs. the other person"
            )
            other_profile = "Nandhitha" if profile == "Ashwin" else "Ashwin"
            col_split1, col_split2 = st.columns(2)
            with col_split1:
                st.write(f"**{profile}:** {split_percentage}%")
            with col_split2:
                st.write(f"**{other_profile}:** {100 - split_percentage}%")

        st.markdown("---")
        # Analyze button
        if st.button("Analyze Nutrition", type="primary"):
            if not food_description.strip():
                st.error("Please enter a food description")
            else:
                with st.spinner("Analyzing nutrition..."):
                    try:
                        # Get nutrition data from Claude
                        nutrition_data = analyze_nutrition(food_description)

                        if split_meal:
                            # Calculate split portions
                            primary_portion = split_percentage / 100.0
                            other_portion = (100 - split_percentage) / 100.0
                            other_profile = "Nandhitha" if profile == "Ashwin" else "Ashwin"

                            # Create record for primary profile
                            primary_nutrition = {
                                key: value * primary_portion
                                for key, value in nutrition_data.items()
                            }
                            primary_record = {
                                'profile': profile,
                                'date': date_str,
                                'food_description': f"{food_description} ({split_percentage}% portion)",
                                **primary_nutrition
                            }

                            # Create record for other profile
                            other_nutrition = {
                                key: value * other_portion
                                for key, value in nutrition_data.items()
                            }
                            other_record = {
                                'profile': other_profile,
                                'date': date_str,
                                'food_description': f"{food_description} ({100 - split_percentage}% portion)",
                                **other_nutrition
                            }

                            # Save both records
                            save_to_csv(primary_record)
                            save_to_csv(other_record)

                            st.success(f"Split meal saved! {profile}: {split_percentage}%, {other_profile}: {100 - split_percentage}%")
                        else:
                            # Single profile entry
                            record = {
                                'profile': profile,
                                'date': date_str,
                                'food_description': food_description,
                                **nutrition_data
                            }
                            save_to_csv(record)
                            st.success("Nutrition data saved successfully!")

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col2:
        st.header("Results")

        # Load data
        df = load_from_csv()

        if not df.empty:
            # Calculate and display daily summary
            summary = calculate_daily_summary(df, profile, date_str)

            st.subheader(f"Daily Summary for {profile} - {date_str}")

            # Get goals for this profile
            goals = DAILY_GOALS.get(profile, {})

            # Display metrics in a grid with goals
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                cal_delta = summary['calories'] - goals.get('calories', 0)
                st.metric(
                    "Calories",
                    f"{summary['calories']:.0f} kcal",
                    delta=f"{cal_delta:+.0f} vs goal ({goals.get('calories', 0)} kcal)",
                    delta_color="normal" if abs(cal_delta) <= 100 else "off"
                )

                prot_delta = summary['protein'] - goals.get('protein', 0)
                st.metric(
                    "Protein",
                    f"{summary['protein']:.1f} g",
                    delta=f"{prot_delta:+.1f}g vs goal ({goals.get('protein', 0)}g)",
                    delta_color="normal"  # More is better: positive=green, negative=red
                )

            with metric_col2:
                carb_delta = summary['carbs'] - goals.get('carbs', 0)
                st.metric(
                    "Carbs",
                    f"{summary['carbs']:.1f} g",
                    delta=f"{carb_delta:+.1f}g vs goal ({goals.get('carbs', 0)}g)",
                    delta_color="normal" if abs(carb_delta) <= 20 else "off"
                )

                fat_delta = summary['fat'] - goals.get('fat', 0)
                st.metric(
                    "Fat",
                    f"{summary['fat']:.1f} g",
                    delta=f"{fat_delta:+.1f}g vs goal ({goals.get('fat', 0)}g)",
                    delta_color="normal" if abs(fat_delta) <= 10 else "off"
                )

            with metric_col3:
                sugar_delta = summary['sugar'] - goals.get('sugar', 0)
                st.metric(
                    "Sugar",
                    f"{summary['sugar']:.1f} g",
                    delta=f"{sugar_delta:+.1f}g vs goal ({goals.get('sugar', 0)}g)",
                    delta_color="inverse"  # Less is better: positive=red, negative=green
                )

                fiber_delta = summary['fiber'] - goals.get('fiber', 0)
                st.metric(
                    "Fiber",
                    f"{summary['fiber']:.1f} g",
                    delta=f"{fiber_delta:+.1f}g vs goal ({goals.get('fiber', 0)}g)",
                    delta_color="normal"  # More is better: positive=green, negative=red
                )

            st.info(f"Total entries today: {summary['entries']}")

            # 7-Day Trend Visualization
            st.markdown("---")
            st.subheader("7-Day Trend")

            # Macro selector
            selected_macro = st.selectbox(
                "Select macro to visualize",
                options=["calories", "protein", "carbs", "fat", "sugar", "fiber"],
                format_func=lambda x: x.capitalize(),
                key="macro_selector"
            )

            # Get 7-day trend data
            trend_data = get_7day_trend(df, profile, date_str)

            if not trend_data.empty and trend_data[selected_macro].sum() > 0:
                # Get goal for selected macro
                macro_goal = DAILY_GOALS.get(profile, {}).get(selected_macro, 0)

                # Create bar chart
                bars = alt.Chart(trend_data).mark_bar().encode(
                    x=alt.X('date:N', title='Date', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y(f'{selected_macro}:Q', title=f'{selected_macro.capitalize()} ({"kcal" if selected_macro == "calories" else "g"})'),
                    tooltip=[
                        alt.Tooltip('date:N', title='Date'),
                        alt.Tooltip(f'{selected_macro}:Q', title=selected_macro.capitalize(), format='.1f')
                    ]
                ).properties(
                    width=600,
                    height=300
                )

                # Create goal line
                goal_line = alt.Chart(pd.DataFrame({'goal': [macro_goal]})).mark_rule(
                    color='green',
                    strokeDash=[5, 5],
                    size=2
                ).encode(
                    y='goal:Q',
                    tooltip=alt.value(f'Goal: {macro_goal}')
                )

                # Combine charts
                chart = (bars + goal_line).configure_mark(
                    color='#FF4B4B'  # Streamlit red for bars
                )

                st.altair_chart(chart, use_container_width=True)
                st.caption(f"🎯 Green dashed line shows daily goal: {macro_goal} {'kcal' if selected_macro == 'calories' else 'g'}")
            else:
                st.info(f"No data available for {selected_macro} in the last 7 days")

            # Show detailed entries for the day
            st.markdown("---")
            st.subheader("Today's Entries")
            daily_entries = df[(df['profile'] == profile) & (df['date'] == date_str)]

            if not daily_entries.empty:
                for idx, row in daily_entries.iterrows():
                    with st.expander(f"Entry: {row['food_description'][:50]}..."):
                        st.write(f"**Food:** {row['food_description']}")
                        st.write(f"**Calories:** {row['calories']:.0f} kcal")
                        st.write(f"**Protein:** {row['protein']:.1f} g | **Carbs:** {row['carbs']:.1f} g | **Fat:** {row['fat']:.1f} g")
                        st.write(f"**Sugar:** {row['sugar']:.1f} g | **Fiber:** {row['fiber']:.1f} g")
            else:
                st.info("No entries for today yet. Add your first meal!")
        else:
            st.info("No data available. Start by adding your first meal!")

    # Optional: Show all data
    with st.expander("View All Data"):
        df_all = load_from_csv()
        if not df_all.empty:
            # Filters
            st.write("**Filters**")
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                # Profile filter - multiselect allows viewing multiple profiles
                selected_profiles = st.multiselect(
                    "Select Profile(s)",
                    options=["Ashwin", "Nandhitha"],
                    default=["Ashwin", "Nandhitha"],
                    key="profile_filter"
                )

            with filter_col2:
                # Date range filter
                all_dates = pd.to_datetime(df_all['date'])
                min_date = all_dates.min().date()
                max_date = all_dates.max().date()

                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="date_filter"
                )

            # Apply filters
            filtered_df = df_all.copy()

            # Filter by profile
            if selected_profiles:
                filtered_df = filtered_df[filtered_df['profile'].isin(selected_profiles)]

            # Filter by date range
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df['date']).dt.date >= start_date) &
                    (pd.to_datetime(filtered_df['date']).dt.date <= end_date)
                ]
            elif date_range:  # Single date selected
                filtered_df = filtered_df[pd.to_datetime(filtered_df['date']).dt.date == date_range]

            st.markdown("---")
            st.write(f"**Showing {len(filtered_df)} of {len(df_all)} entries**")

            # Initialize session state for editing
            if 'editing_entry' not in st.session_state:
                st.session_state.editing_entry = None

            # Create header
            col_date, col_profile, col_food, col_cals, col_protein, col_carbs, col_fat, col_sugar, col_fiber, col_edit, col_delete = st.columns([0.8, 0.8, 2.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5])
            with col_date:
                st.write("**Date**")
            with col_profile:
                st.write("**Profile**")
            with col_food:
                st.write("**Food**")
            with col_cals:
                st.write("**Cal**")
            with col_protein:
                st.write("**Prot**")
            with col_carbs:
                st.write("**Carbs**")
            with col_fat:
                st.write("**Fat**")
            with col_sugar:
                st.write("**Sugar**")
            with col_fiber:
                st.write("**Fiber**")
            with col_edit:
                st.write("**Edit**")
            with col_delete:
                st.write("**Del**")

            st.markdown("---")

            # Display each row (filtered)
            for idx, row in filtered_df.iterrows():
                is_editing = st.session_state.editing_entry == idx

                if is_editing:
                    # Edit mode
                    col_date, col_profile, col_food, col_cals, col_protein, col_carbs, col_fat, col_sugar, col_fiber, col_edit, col_delete = st.columns([0.8, 0.8, 2.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5])

                    with col_date:
                        st.text(row['date'])
                    with col_profile:
                        st.text(row['profile'])
                    with col_food:
                        new_food = st.text_input("Food", value=row['food_description'], key=f"edit_food_{idx}", label_visibility="collapsed")
                    with col_cals:
                        new_cals = st.number_input("Cal", value=float(row['calories']), key=f"edit_cal_{idx}", label_visibility="collapsed", step=1.0, format="%.0f")
                    with col_protein:
                        new_protein = st.number_input("Prot", value=float(row['protein']), key=f"edit_prot_{idx}", label_visibility="collapsed", step=0.1, format="%.1f")
                    with col_carbs:
                        new_carbs = st.number_input("Carbs", value=float(row['carbs']), key=f"edit_carbs_{idx}", label_visibility="collapsed", step=0.1, format="%.1f")
                    with col_fat:
                        new_fat = st.number_input("Fat", value=float(row['fat']), key=f"edit_fat_{idx}", label_visibility="collapsed", step=0.1, format="%.1f")
                    with col_sugar:
                        new_sugar = st.number_input("Sugar", value=float(row['sugar']), key=f"edit_sugar_{idx}", label_visibility="collapsed", step=0.1, format="%.1f")
                    with col_fiber:
                        new_fiber = st.number_input("Fiber", value=float(row['fiber']), key=f"edit_fiber_{idx}", label_visibility="collapsed", step=0.1, format="%.1f")
                    with col_edit:
                        if st.button("💾", key=f"save_{idx}", help="Save changes"):
                            # Check if food description changed
                            food_changed = new_food != row['food_description']

                            if food_changed:
                                # Re-analyze with LLM
                                with st.spinner("Re-analyzing nutrition..."):
                                    try:
                                        new_nutrition = analyze_nutrition(new_food)
                                        updated_data = {
                                            'food_description': new_food,
                                            'calories': new_nutrition['calories'],
                                            'protein': new_nutrition['protein'],
                                            'carbs': new_nutrition['carbs'],
                                            'fat': new_nutrition['fat'],
                                            'sugar': new_nutrition['sugar'],
                                            'fiber': new_nutrition['fiber']
                                        }
                                        update_entry(idx, updated_data)
                                        st.session_state.editing_entry = None
                                        st.success("Entry updated with new LLM analysis!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error re-analyzing: {str(e)}")
                            else:
                                # Just update the manual values
                                updated_data = {
                                    'calories': new_cals,
                                    'protein': new_protein,
                                    'carbs': new_carbs,
                                    'fat': new_fat,
                                    'sugar': new_sugar,
                                    'fiber': new_fiber
                                }
                                update_entry(idx, updated_data)
                                st.session_state.editing_entry = None
                                st.success("Entry updated!")
                                st.rerun()
                    with col_delete:
                        if st.button("✖️", key=f"cancel_{idx}", help="Cancel editing"):
                            st.session_state.editing_entry = None
                            st.rerun()
                else:
                    # View mode
                    col_date, col_profile, col_food, col_cals, col_protein, col_carbs, col_fat, col_sugar, col_fiber, col_edit, col_delete = st.columns([0.8, 0.8, 2.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5])

                    with col_date:
                        st.text(row['date'])
                    with col_profile:
                        st.text(row['profile'])
                    with col_food:
                        food_text = row['food_description'][:30]
                        if len(row['food_description']) > 30:
                            food_text += "..."
                        st.text(food_text)
                    with col_cals:
                        st.text(f"{row['calories']:.0f}")
                    with col_protein:
                        st.text(f"{row['protein']:.1f}g")
                    with col_carbs:
                        st.text(f"{row['carbs']:.1f}g")
                    with col_fat:
                        st.text(f"{row['fat']:.1f}g")
                    with col_sugar:
                        st.text(f"{row['sugar']:.1f}g")
                    with col_fiber:
                        st.text(f"{row['fiber']:.1f}g")
                    with col_edit:
                        if st.button("✏️", key=f"edit_{idx}", help="Edit this entry"):
                            st.session_state.editing_entry = idx
                            st.rerun()
                    with col_delete:
                        if st.button("🗑️", key=f"delete_all_{idx}", help="Delete this entry"):
                            delete_entry(idx)
                            st.rerun()
        else:
            st.info("No data available yet")


if __name__ == "__main__":
    main()
