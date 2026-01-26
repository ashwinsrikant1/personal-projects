import streamlit as st
import anthropic
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import altair as alt
import base64
from zoneinfo import ZoneInfo

# Load environment variables from .env file
load_dotenv()

# Hardcoded API key placeholder (fallback)
ANTHROPIC_API_KEY = "# TODO: Add your Anthropic API key here"

def get_local_now():
    """
    Get current datetime in local timezone.
    Uses PST by default, or local system timezone if explicitly running locally.

    Returns:
        datetime: Current datetime in PST or local timezone
    """
    try:
        # Check if we're running in a cloud environment (like Streamlit Cloud)
        # by detecting if the system timezone is UTC
        local_now = datetime.now().astimezone()
        tz_name = local_now.strftime('%Z')

        # If system is UTC (typical for cloud deployments), use PST instead
        if tz_name == 'UTC':
            pst_tz = ZoneInfo("America/Los_Angeles")
            return datetime.now(pst_tz)
        else:
            # Use local system timezone (for local development)
            return local_now
    except:
        # Fallback to PST (America/Los_Angeles)
        try:
            pst_tz = ZoneInfo("America/Los_Angeles")
            return datetime.now(pst_tz)
        except:
            # Ultimate fallback: use naive datetime
            return datetime.now()

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

def analyze_nutrition(food_description: str, image_data: bytes = None, image_type: str = None) -> dict:
    """
    Analyze food description and/or image using Claude API and return nutritional information.

    Args:
        food_description: Text description of food consumed
        image_data: Optional image bytes (from uploaded file or camera)
        image_type: Optional MIME type of the image (e.g., 'image/jpeg', 'image/png')

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

    base_prompt = """Analyze the provided food information and provide detailed nutritional information.
Return ONLY a JSON object with the following fields (all numeric values):
- calories (kcal)
- protein (g)
- carbs (g)
- fat (g)
- sugar (g) - IMPORTANT: This should be ADDED SUGAR only, NOT total sugar. Do not include natural sugars from fruits, milk, etc.
- fiber (g)

"""

    # Build message content based on whether we have an image
    content = []

    if image_data and image_type:
        # Add image to content
        image_base64 = base64.standard_b64encode(image_data).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_type,
                "data": image_base64
            }
        })

        if food_description.strip():
            prompt = base_prompt + f"Food description: {food_description}\n\nAlso analyze the attached image which may show nutritional information or the food itself. Use both the description and image to provide accurate nutrition data.\n\nReturn only the JSON object, no other text."
        else:
            prompt = base_prompt + "Analyze the attached image which may show nutritional information (nutrition label) or the food itself. Extract or estimate the nutrition data from the image.\n\nReturn only the JSON object, no other text."
    else:
        prompt = base_prompt + f"Food description: {food_description}\n\nReturn only the JSON object, no other text."

    content.append({"type": "text", "text": prompt})

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": content}
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
        end_date = get_local_now().strftime("%Y-%m-%d")

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

    # Show current timezone info
    current_tz = get_local_now()
    tz_name = current_tz.strftime('%Z')  # Timezone name (e.g., PST, EST)
    tz_offset = current_tz.strftime('%z')  # Timezone offset (e.g., -0800)
    st.caption(f"🕐 Using timezone: {tz_name} (UTC{tz_offset[:3]}:{tz_offset[3:]})")

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
            value=get_local_now().date()
        )
        date_str = selected_date.strftime("%Y-%m-%d")

        # Initialize session state for food description if not exists
        if 'food_description' not in st.session_state:
            st.session_state.food_description = ""

        # Food description text area with clear functionality
        food_description = st.text_area(
            "Food Description",
            value=st.session_state.food_description,
            placeholder="E.g., 2 scrambled eggs, 1 slice of whole wheat toast with butter, 1 cup of coffee with milk",
            height=150,
            key="food_input"
        )

        # Update session state when user types
        st.session_state.food_description = food_description

        # Clear button
        if st.button("Clear Description", help="Clear the food description text"):
            st.session_state.food_description = ""
            st.rerun()

        # Image input section (optional)
        st.markdown("**Add Photo (Optional)**")
        st.caption("Upload a photo of nutritional info or food. Works with text description or standalone.")

        # Two options: file upload (desktop) or camera (mobile)
        image_input_method = st.radio(
            "Image input method",
            options=["Upload Photo", "Take Photo (Camera)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_image = None
        image_data = None
        image_type = None

        if image_input_method == "Upload Photo":
            uploaded_image = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "webp", "gif"],
                help="Upload a photo of the nutrition label or the food itself",
                label_visibility="collapsed"
            )
        else:
            uploaded_image = st.camera_input(
                "Take a photo",
                help="Take a photo of the nutrition label or food",
                label_visibility="collapsed"
            )

        # Process uploaded image if present
        if uploaded_image is not None:
            image_data = uploaded_image.getvalue()
            # Determine image type
            if uploaded_image.type:
                image_type = uploaded_image.type
            else:
                # Fallback based on file extension or default to jpeg
                image_type = "image/jpeg"

            # Show preview
            st.image(uploaded_image, caption="Attached image", width=200)

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
            if not food_description.strip() and image_data is None:
                st.error("Please enter a food description or attach an image")
            else:
                with st.spinner("Analyzing nutrition..."):
                    try:
                        # Get nutrition data from Claude (with optional image)
                        nutrition_data = analyze_nutrition(food_description, image_data, image_type)

                        # Determine the description to save
                        if food_description.strip():
                            saved_description = food_description
                            if image_data:
                                saved_description += " (with image)"
                        else:
                            # Image-only submission
                            saved_description = "[Analyzed from image]"

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
                                'food_description': f"{saved_description} ({split_percentage}% portion)",
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
                                'food_description': f"{saved_description} ({100 - split_percentage}% portion)",
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
                                'food_description': saved_description,
                                **nutrition_data
                            }
                            save_to_csv(record)
                            st.success("Nutrition data saved successfully!")

                        # Clear the food description after successful save
                        st.session_state.food_description = ""

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

    # Export/Import Data section
    with st.expander("Export / Import Data"):
        st.markdown("### Export Data")
        st.write("Download all your nutrition data as a CSV file for backup.")

        df_export = load_from_csv()
        if not df_export.empty:
            # Convert DataFrame to CSV string
            csv_data = df_export.to_csv(index=False)

            # Create download button
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"nutrition_data_backup_{get_local_now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all nutrition data as CSV"
            )
        else:
            st.info("No data available to export")

        st.markdown("---")
        st.markdown("### Import Data")
        st.write("Upload a previously exported CSV file to restore your data.")
        st.warning("⚠️ Importing will **add** the uploaded data to your existing data. If you want to replace all data, delete entries first.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="import_csv")

        if uploaded_file is not None:
            try:
                # Read uploaded CSV
                df_uploaded = pd.read_csv(uploaded_file)

                # Validate columns
                required_columns = ['profile', 'date', 'food_description', 'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber']
                if not all(col in df_uploaded.columns for col in required_columns):
                    st.error(f"Invalid CSV format. Required columns: {', '.join(required_columns)}")
                else:
                    st.success(f"File loaded successfully! Found {len(df_uploaded)} entries.")

                    # Show preview
                    st.write("**Preview (first 5 rows):**")
                    st.dataframe(df_uploaded.head())

                    # Import button
                    if st.button("Import Data", type="primary"):
                        # Load existing data
                        df_existing = load_from_csv()

                        # Combine with uploaded data
                        if not df_existing.empty:
                            df_combined = pd.concat([df_existing, df_uploaded], ignore_index=True)
                        else:
                            df_combined = df_uploaded

                        # Save combined data
                        df_combined.to_csv("nutrition_data.csv", index=False)

                        st.success(f"Successfully imported {len(df_uploaded)} entries!")
                        st.rerun()

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

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

            # Mobile-friendly card layout using expanders
            for idx, row in filtered_df.iterrows():
                is_editing = st.session_state.editing_entry == idx

                # Create summary for expander header
                food_preview = row['food_description'][:40]
                if len(row['food_description']) > 40:
                    food_preview += "..."
                header = f"{row['date']} | {row['profile']} | {row['calories']:.0f} cal | {food_preview}"

                with st.expander(header):
                    if is_editing:
                        # Edit mode
                        st.write("**Edit Entry**")
                        new_food = st.text_area("Food Description", value=row['food_description'], key=f"edit_food_{idx}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            new_cals = st.number_input("Calories", value=float(row['calories']), key=f"edit_cal_{idx}", step=1.0, format="%.0f")
                            new_protein = st.number_input("Protein (g)", value=float(row['protein']), key=f"edit_prot_{idx}", step=0.1, format="%.1f")
                        with col2:
                            new_carbs = st.number_input("Carbs (g)", value=float(row['carbs']), key=f"edit_carbs_{idx}", step=0.1, format="%.1f")
                            new_fat = st.number_input("Fat (g)", value=float(row['fat']), key=f"edit_fat_{idx}", step=0.1, format="%.1f")
                        with col3:
                            new_sugar = st.number_input("Sugar (g)", value=float(row['sugar']), key=f"edit_sugar_{idx}", step=0.1, format="%.1f")
                            new_fiber = st.number_input("Fiber (g)", value=float(row['fiber']), key=f"edit_fiber_{idx}", step=0.1, format="%.1f")

                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("Save", key=f"save_{idx}", type="primary"):
                                food_changed = new_food != row['food_description']
                                if food_changed:
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
                        with btn_col2:
                            if st.button("Cancel", key=f"cancel_{idx}"):
                                st.session_state.editing_entry = None
                                st.rerun()
                    else:
                        # View mode
                        st.write(f"**Food:** {row['food_description']}")
                        st.write(f"**Calories:** {row['calories']:.0f} kcal")
                        st.write(f"**Protein:** {row['protein']:.1f}g | **Carbs:** {row['carbs']:.1f}g | **Fat:** {row['fat']:.1f}g")
                        st.write(f"**Sugar:** {row['sugar']:.1f}g | **Fiber:** {row['fiber']:.1f}g")

                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("Edit", key=f"edit_{idx}"):
                                st.session_state.editing_entry = idx
                                st.rerun()
                        with btn_col2:
                            if st.button("Delete", key=f"delete_all_{idx}"):
                                delete_entry(idx)
                                st.rerun()
        else:
            st.info("No data available yet")


if __name__ == "__main__":
    main()
