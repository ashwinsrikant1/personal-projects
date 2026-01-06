# Nutrition Tracker

An AI-powered nutrition tracking application that uses Claude to analyze food descriptions and track macronutrients for multiple users.

## Features

- **Multi-Profile Support**: Track nutrition for Ashwin and Nandhitha
- **AI-Powered Analysis**: Uses Claude Sonnet 4.5 to analyze food descriptions and extract nutritional information
- **Extended Macros**: Tracks calories, protein, carbs, fat, sugar, and fiber
- **Daily Summaries**: View aggregated nutrition data for each day
- **CSV Storage**: All data is saved locally in CSV format
- **Interactive UI**: Built with Streamlit for a clean, responsive interface

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your Anthropic API key:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

You can get an API key from [Anthropic Console](https://console.anthropic.com/settings/keys).

### 3. Run the Application

```bash
streamlit run nutrition_tracker.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

1. **Select Profile**: Choose between Ashwin or Nandhitha
2. **Pick a Date**: Select the date for the entry (defaults to today)
3. **Describe Your Food**: Enter a description of what you ate
   - Example: "2 scrambled eggs, 1 slice of whole wheat toast with butter, 1 cup of coffee with milk"
4. **Analyze**: Click the "Analyze Nutrition" button
5. **View Results**: See the nutritional breakdown and daily summary on the right

### Daily Summary

The app displays:
- Total calories, protein, carbs, fat, sugar, and fiber for the day
- Number of entries logged
- Detailed view of each food entry

## Data Storage

All nutrition data is stored in `nutrition_data.csv` in the same directory as the application. The CSV includes:
- Profile name
- Date
- Food description
- Nutritional values (calories, protein, carbs, fat, sugar, fiber)

## Running Tests

The project includes a comprehensive test suite using pytest:

```bash
# Run all tests
pytest test_nutrition_tracker.py

# Run with verbose output
pytest test_nutrition_tracker.py -v

# Run with coverage report
pytest test_nutrition_tracker.py --cov=nutrition_tracker
```

### Test Coverage

The test suite includes:
- **API Integration Tests**: Mock Anthropic API calls and response parsing
- **CSV Operations**: Testing save and load functionality
- **Daily Summary Calculations**: Aggregation logic for multiple entries
- **Error Handling**: Missing API keys, invalid JSON, missing fields
- **Edge Cases**: Empty datasets, multiple profiles, different dates

## Project Structure

```
nutrition-tracker/
├── nutrition_tracker.py      # Main Streamlit application
├── test_nutrition_tracker.py # Pytest test suite
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── README.md                # This file
└── nutrition_data.csv       # Data storage (created on first use)
```

## Technical Details

### API Integration

- **Model**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **Max Tokens**: 1024
- **Response Format**: JSON with nutritional values

### Data Schema

```python
{
    'profile': str,              # User profile name
    'date': str,                 # YYYY-MM-DD format
    'food_description': str,     # Text description of food
    'calories': float,           # Kilocalories
    'protein': float,            # Grams
    'carbs': float,              # Grams
    'fat': float,                # Grams
    'sugar': float,              # Grams
    'fiber': float               # Grams
}
```

## Troubleshooting

### API Key Issues

If you see "Please set your Anthropic API key":
1. Ensure `.env` file exists in the project directory
2. Verify the API key is correctly set in `.env`
3. Restart the Streamlit application

### CSV File Location

The `nutrition_data.csv` file is created in the directory where you run the `streamlit run` command. Make sure you have write permissions in that directory.

### Import Errors

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Contributing

This is a personal nutrition tracking application. Feel free to fork and customize for your own use.

## License

This project is for personal use. The Anthropic API usage is subject to Anthropic's terms of service.
