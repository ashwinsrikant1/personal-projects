import pytest
import pandas as pd
import json
import os
from unittest.mock import Mock, patch, MagicMock
from nutrition_tracker import (
    analyze_nutrition,
    save_to_csv,
    load_from_csv,
    calculate_daily_summary
)


@pytest.fixture
def sample_nutrition_data():
    """Sample nutrition data for testing"""
    return {
        'calories': 450,
        'protein': 25.5,
        'carbs': 35.2,
        'fat': 18.3,
        'sugar': 5.1,
        'fiber': 4.2
    }


@pytest.fixture
def sample_record(sample_nutrition_data):
    """Sample complete record for testing"""
    return {
        'profile': 'Ashwin',
        'date': '2026-01-06',
        'food_description': '2 scrambled eggs with toast',
        **sample_nutrition_data
    }


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file path"""
    return str(tmp_path / "test_nutrition.csv")


@pytest.fixture
def mock_anthropic_response(sample_nutrition_data):
    """Mock Anthropic API response"""
    mock_message = Mock()
    mock_content = Mock()
    mock_content.text = json.dumps(sample_nutrition_data)
    mock_message.content = [mock_content]
    return mock_message


class TestAnalyzeNutrition:
    """Tests for analyze_nutrition function"""

    def test_analyze_nutrition_success(self, mock_anthropic_response, sample_nutrition_data):
        """Test successful nutrition analysis"""
        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
                result = analyze_nutrition("2 scrambled eggs with toast")

            assert result == sample_nutrition_data
            assert 'calories' in result
            assert 'protein' in result
            assert 'carbs' in result
            assert 'fat' in result
            assert 'sugar' in result
            assert 'fiber' in result

    def test_analyze_nutrition_missing_api_key(self):
        """Test error when API key is missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Please set your Anthropic API key"):
                analyze_nutrition("test food")

    def test_analyze_nutrition_invalid_json_response(self):
        """Test error handling for invalid JSON response"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "This is not valid JSON"
        mock_message.content = [mock_content]

        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
                with pytest.raises(ValueError, match="Failed to parse nutrition data"):
                    analyze_nutrition("test food")

    def test_analyze_nutrition_missing_required_fields(self):
        """Test error when required fields are missing from response"""
        incomplete_data = {'calories': 100, 'protein': 10}  # Missing other fields
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps(incomplete_data)
        mock_message.content = [mock_content]

        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
                with pytest.raises(ValueError, match="Missing required field"):
                    analyze_nutrition("test food")

    def test_analyze_nutrition_api_error(self):
        """Test error handling for API errors"""
        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("API Error")
            mock_anthropic.return_value = mock_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
                with pytest.raises(Exception, match="Error analyzing nutrition"):
                    analyze_nutrition("test food")

    def test_analyze_nutrition_uses_correct_model(self, mock_anthropic_response):
        """Test that the correct Claude model is used"""
        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key'}):
                analyze_nutrition("test food")

            # Verify the correct model was used
            call_args = mock_client.messages.create.call_args
            assert call_args[1]['model'] == 'claude-sonnet-4-5-20250929'


class TestSaveToCsv:
    """Tests for save_to_csv function"""

    def test_save_to_csv_new_file(self, sample_record, temp_csv_file):
        """Test saving to a new CSV file"""
        save_to_csv(sample_record, temp_csv_file)

        assert os.path.exists(temp_csv_file)

        df = pd.read_csv(temp_csv_file)
        assert len(df) == 1
        assert df.iloc[0]['profile'] == 'Ashwin'
        assert df.iloc[0]['calories'] == 450
        assert df.iloc[0]['protein'] == 25.5

    def test_save_to_csv_append_to_existing(self, sample_record, temp_csv_file):
        """Test appending to an existing CSV file"""
        # Save first record
        save_to_csv(sample_record, temp_csv_file)

        # Save second record
        second_record = sample_record.copy()
        second_record['profile'] = 'Nandhitha'
        second_record['calories'] = 350
        save_to_csv(second_record, temp_csv_file)

        df = pd.read_csv(temp_csv_file)
        assert len(df) == 2
        assert df.iloc[0]['profile'] == 'Ashwin'
        assert df.iloc[1]['profile'] == 'Nandhitha'

    def test_save_to_csv_multiple_entries(self, sample_record, temp_csv_file):
        """Test saving multiple entries"""
        for i in range(5):
            record = sample_record.copy()
            record['calories'] = 100 + (i * 50)
            save_to_csv(record, temp_csv_file)

        df = pd.read_csv(temp_csv_file)
        assert len(df) == 5

    def test_save_to_csv_preserves_all_fields(self, sample_record, temp_csv_file):
        """Test that all fields are preserved when saving"""
        save_to_csv(sample_record, temp_csv_file)

        df = pd.read_csv(temp_csv_file)
        row = df.iloc[0]

        assert row['profile'] == sample_record['profile']
        assert row['date'] == sample_record['date']
        assert row['food_description'] == sample_record['food_description']
        assert row['calories'] == sample_record['calories']
        assert row['protein'] == sample_record['protein']
        assert row['carbs'] == sample_record['carbs']
        assert row['fat'] == sample_record['fat']
        assert row['sugar'] == sample_record['sugar']
        assert row['fiber'] == sample_record['fiber']


class TestLoadFromCsv:
    """Tests for load_from_csv function"""

    def test_load_from_csv_existing_file(self, sample_record, temp_csv_file):
        """Test loading from an existing CSV file"""
        save_to_csv(sample_record, temp_csv_file)
        df = load_from_csv(temp_csv_file)

        assert len(df) == 1
        assert df.iloc[0]['profile'] == 'Ashwin'

    def test_load_from_csv_nonexistent_file(self, temp_csv_file):
        """Test loading from a nonexistent file returns empty DataFrame"""
        df = load_from_csv(temp_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == [
            'profile', 'date', 'food_description',
            'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'
        ]

    def test_load_from_csv_multiple_records(self, sample_record, temp_csv_file):
        """Test loading multiple records"""
        for i in range(3):
            record = sample_record.copy()
            record['calories'] = 100 + (i * 50)
            save_to_csv(record, temp_csv_file)

        df = load_from_csv(temp_csv_file)
        assert len(df) == 3


class TestCalculateDailySummary:
    """Tests for calculate_daily_summary function"""

    def test_calculate_daily_summary_single_entry(self, sample_record, temp_csv_file):
        """Test daily summary with a single entry"""
        save_to_csv(sample_record, temp_csv_file)
        df = load_from_csv(temp_csv_file)

        summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')

        assert summary['calories'] == 450
        assert summary['protein'] == 25.5
        assert summary['carbs'] == 35.2
        assert summary['fat'] == 18.3
        assert summary['sugar'] == 5.1
        assert summary['fiber'] == 4.2
        assert summary['entries'] == 1

    def test_calculate_daily_summary_multiple_entries(self, sample_record, temp_csv_file):
        """Test daily summary with multiple entries"""
        for i in range(3):
            save_to_csv(sample_record, temp_csv_file)

        df = load_from_csv(temp_csv_file)
        summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')

        assert summary['calories'] == 450 * 3
        assert summary['protein'] == 25.5 * 3
        assert summary['entries'] == 3

    def test_calculate_daily_summary_different_profiles(self, sample_record, temp_csv_file):
        """Test daily summary filters by profile correctly"""
        # Add entries for Ashwin
        save_to_csv(sample_record, temp_csv_file)

        # Add entry for Nandhitha
        nandhitha_record = sample_record.copy()
        nandhitha_record['profile'] = 'Nandhitha'
        nandhitha_record['calories'] = 300
        save_to_csv(nandhitha_record, temp_csv_file)

        df = load_from_csv(temp_csv_file)

        ashwin_summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')
        nandhitha_summary = calculate_daily_summary(df, 'Nandhitha', '2026-01-06')

        assert ashwin_summary['calories'] == 450
        assert ashwin_summary['entries'] == 1
        assert nandhitha_summary['calories'] == 300
        assert nandhitha_summary['entries'] == 1

    def test_calculate_daily_summary_different_dates(self, sample_record, temp_csv_file):
        """Test daily summary filters by date correctly"""
        # Add entry for Jan 6
        save_to_csv(sample_record, temp_csv_file)

        # Add entry for Jan 7
        jan7_record = sample_record.copy()
        jan7_record['date'] = '2026-01-07'
        jan7_record['calories'] = 300
        save_to_csv(jan7_record, temp_csv_file)

        df = load_from_csv(temp_csv_file)

        jan6_summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')
        jan7_summary = calculate_daily_summary(df, 'Ashwin', '2026-01-07')

        assert jan6_summary['calories'] == 450
        assert jan6_summary['entries'] == 1
        assert jan7_summary['calories'] == 300
        assert jan7_summary['entries'] == 1

    def test_calculate_daily_summary_no_data(self):
        """Test daily summary with no data"""
        df = pd.DataFrame(columns=[
            'profile', 'date', 'food_description',
            'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'
        ])

        summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')

        assert summary['calories'] == 0
        assert summary['protein'] == 0
        assert summary['carbs'] == 0
        assert summary['fat'] == 0
        assert summary['sugar'] == 0
        assert summary['fiber'] == 0
        assert summary['entries'] == 0

    def test_calculate_daily_summary_wrong_profile(self, sample_record, temp_csv_file):
        """Test daily summary with non-matching profile"""
        save_to_csv(sample_record, temp_csv_file)
        df = load_from_csv(temp_csv_file)

        summary = calculate_daily_summary(df, 'NonExistentProfile', '2026-01-06')

        assert summary['entries'] == 0
        assert summary['calories'] == 0
