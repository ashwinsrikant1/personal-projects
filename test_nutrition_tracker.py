import pytest
import pandas as pd
import json
import os
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import contextmanager
from nutrition_tracker import (
    analyze_nutrition,
    save_to_csv,
    load_from_csv,
    update_entry,
    delete_entry,
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
def mock_anthropic_response(sample_nutrition_data):
    """Mock Anthropic API response"""
    mock_message = Mock()
    mock_content = Mock()
    mock_content.text = json.dumps(sample_nutrition_data)
    mock_message.content = [mock_content]
    return mock_message


@pytest.fixture
def mock_db():
    """Mock database connection context manager."""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

    @contextmanager
    def fake_get_db_connection():
        yield mock_conn

    with patch('nutrition_tracker.get_db_connection', fake_get_db_connection):
        yield mock_conn, mock_cursor


class TestAnalyzeNutrition:
    """Tests for analyze_nutrition function"""

    def test_analyze_nutrition_success(self, mock_anthropic_response, sample_nutrition_data):
        """Test successful nutrition analysis with both models returning same data"""
        mock_gemini_response = Mock()
        mock_gemini_response.text = json.dumps(sample_nutrition_data)

        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic, \
             patch('nutrition_tracker.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            mock_gemini_client = Mock()
            mock_gemini_client.models.generate_content.return_value = mock_gemini_response
            mock_genai.return_value = mock_gemini_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key', 'GEMINI_API_KEY': 'test_gemini_key'}):
                result = analyze_nutrition("2 scrambled eggs with toast")

            # Both models return same data, so average equals the original
            assert result == sample_nutrition_data
            assert 'calories' in result
            assert 'protein' in result
            assert 'carbs' in result
            assert 'fat' in result
            assert 'sugar' in result
            assert 'fiber' in result

    def test_analyze_nutrition_missing_api_key(self):
        """Test error when both API keys are missing"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception, match="Both models failed"):
                analyze_nutrition("test food")

    def test_analyze_nutrition_invalid_json_response(self):
        """Test error handling for invalid JSON response from both models"""
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "This is not valid JSON"
        mock_message.content = [mock_content]

        mock_gemini_response = Mock()
        mock_gemini_response.text = "This is not valid JSON either"

        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic, \
             patch('nutrition_tracker.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            mock_gemini_client = Mock()
            mock_gemini_client.models.generate_content.return_value = mock_gemini_response
            mock_genai.return_value = mock_gemini_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key', 'GEMINI_API_KEY': 'test_gemini_key'}):
                with pytest.raises(Exception, match="Both models failed"):
                    analyze_nutrition("test food")

    def test_analyze_nutrition_missing_required_fields(self):
        """Test error when required fields are missing from both model responses"""
        incomplete_data = {'calories': 100, 'protein': 10}  # Missing other fields
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps(incomplete_data)
        mock_message.content = [mock_content]

        mock_gemini_response = Mock()
        mock_gemini_response.text = json.dumps(incomplete_data)

        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic, \
             patch('nutrition_tracker.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            mock_gemini_client = Mock()
            mock_gemini_client.models.generate_content.return_value = mock_gemini_response
            mock_genai.return_value = mock_gemini_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key', 'GEMINI_API_KEY': 'test_gemini_key'}):
                with pytest.raises(Exception, match="Both models failed"):
                    analyze_nutrition("test food")

    def test_analyze_nutrition_api_error(self):
        """Test error handling when both APIs fail"""
        with patch('nutrition_tracker.anthropic.Anthropic') as mock_anthropic, \
             patch('nutrition_tracker.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("API Error")
            mock_anthropic.return_value = mock_client

            mock_gemini_client = Mock()
            mock_gemini_client.models.generate_content.side_effect = Exception("Gemini API Error")
            mock_genai.return_value = mock_gemini_client

            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_api_key', 'GEMINI_API_KEY': 'test_gemini_key'}):
                with pytest.raises(Exception, match="Both models failed"):
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
    """Tests for save_to_csv function (now DB-backed)"""

    def test_save_inserts_record(self, sample_record, mock_db):
        """Test saving inserts a record into the database"""
        mock_conn, mock_cursor = mock_db
        save_to_csv(sample_record)

        mock_cursor.execute.assert_called_once()
        args = mock_cursor.execute.call_args
        assert "INSERT INTO nutrition_entries" in args[0][0]
        assert args[0][1] == (
            'Ashwin', '2026-01-06', '2 scrambled eggs with toast',
            450, 25.5, 35.2, 18.3, 5.1, 4.2
        )

    def test_save_multiple_records(self, sample_record, mock_db):
        """Test saving multiple records executes multiple inserts"""
        mock_conn, mock_cursor = mock_db

        for i in range(5):
            record = sample_record.copy()
            record['calories'] = 100 + (i * 50)
            save_to_csv(record)

        assert mock_cursor.execute.call_count == 5

    def test_save_preserves_all_fields(self, sample_record, mock_db):
        """Test that all fields are passed to the INSERT"""
        mock_conn, mock_cursor = mock_db
        save_to_csv(sample_record)

        args = mock_cursor.execute.call_args[0][1]
        assert args[0] == sample_record['profile']
        assert args[1] == sample_record['date']
        assert args[2] == sample_record['food_description']
        assert args[3] == sample_record['calories']
        assert args[4] == sample_record['protein']
        assert args[5] == sample_record['carbs']
        assert args[6] == sample_record['fat']
        assert args[7] == sample_record['sugar']
        assert args[8] == sample_record['fiber']


class TestLoadFromCsv:
    """Tests for load_from_csv function (now DB-backed)"""

    def test_load_returns_dataframe_with_data(self):
        """Test loading returns a DataFrame with data from DB"""
        expected_df = pd.DataFrame({
            'id': [1],
            'profile': ['Ashwin'],
            'date': ['2026-01-06'],
            'food_description': ['2 scrambled eggs with toast'],
            'calories': [450.0],
            'protein': [25.5],
            'carbs': [35.2],
            'fat': [18.3],
            'sugar': [5.1],
            'fiber': [4.2]
        })

        @contextmanager
        def fake_conn():
            yield MagicMock()

        with patch('nutrition_tracker.get_db_connection', fake_conn), \
             patch('nutrition_tracker.pd.read_sql_query', return_value=expected_df):
            df = load_from_csv()

        assert len(df) == 1
        assert df.iloc[0]['profile'] == 'Ashwin'
        assert df.index[0] == 1  # id is used as index

    def test_load_returns_empty_dataframe_when_no_data(self):
        """Test loading returns empty DataFrame when DB has no rows"""
        empty_df = pd.DataFrame(columns=[
            'id', 'profile', 'date', 'food_description',
            'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'
        ])

        @contextmanager
        def fake_conn():
            yield MagicMock()

        with patch('nutrition_tracker.get_db_connection', fake_conn), \
             patch('nutrition_tracker.pd.read_sql_query', return_value=empty_df):
            df = load_from_csv()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == [
            'profile', 'date', 'food_description',
            'calories', 'protein', 'carbs', 'fat', 'sugar', 'fiber'
        ]

    def test_load_multiple_records(self):
        """Test loading multiple records"""
        expected_df = pd.DataFrame({
            'id': [1, 2, 3],
            'profile': ['Ashwin', 'Ashwin', 'Ashwin'],
            'date': ['2026-01-06', '2026-01-06', '2026-01-06'],
            'food_description': ['eggs', 'toast', 'coffee'],
            'calories': [150.0, 200.0, 100.0],
            'protein': [12.0, 5.0, 1.0],
            'carbs': [1.0, 30.0, 5.0],
            'fat': [10.0, 3.0, 2.0],
            'sugar': [0.0, 2.0, 3.0],
            'fiber': [0.0, 2.0, 0.0]
        })

        @contextmanager
        def fake_conn():
            yield MagicMock()

        with patch('nutrition_tracker.get_db_connection', fake_conn), \
             patch('nutrition_tracker.pd.read_sql_query', return_value=expected_df):
            df = load_from_csv()

        assert len(df) == 3


class TestUpdateEntry:
    """Tests for update_entry function (now DB-backed)"""

    def test_update_entry(self, mock_db):
        """Test updating an entry executes correct SQL"""
        mock_conn, mock_cursor = mock_db
        update_entry(42, {'calories': 500.0, 'protein': 30.0})

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "UPDATE nutrition_entries" in sql
        assert "WHERE id = %s" in sql
        # Last value should be the id
        values = mock_cursor.execute.call_args[0][1]
        assert values[-1] == 42

    def test_update_entry_empty_data(self, mock_db):
        """Test that updating with empty dict does nothing"""
        mock_conn, mock_cursor = mock_db
        update_entry(42, {})
        mock_cursor.execute.assert_not_called()


class TestDeleteEntry:
    """Tests for delete_entry function (now DB-backed)"""

    def test_delete_entry(self, mock_db):
        """Test deleting an entry executes correct SQL"""
        mock_conn, mock_cursor = mock_db
        delete_entry(42)

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "DELETE FROM nutrition_entries" in sql
        assert "WHERE id = %s" in sql
        assert mock_cursor.execute.call_args[0][1] == (42,)


class TestCalculateDailySummary:
    """Tests for calculate_daily_summary function"""

    def test_calculate_daily_summary_single_entry(self, sample_nutrition_data):
        """Test daily summary with a single entry"""
        df = pd.DataFrame([{
            'profile': 'Ashwin',
            'date': '2026-01-06',
            'food_description': '2 scrambled eggs with toast',
            **sample_nutrition_data
        }])

        summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')

        assert summary['calories'] == 450
        assert summary['protein'] == 25.5
        assert summary['carbs'] == 35.2
        assert summary['fat'] == 18.3
        assert summary['sugar'] == 5.1
        assert summary['fiber'] == 4.2
        assert summary['entries'] == 1

    def test_calculate_daily_summary_multiple_entries(self, sample_nutrition_data):
        """Test daily summary with multiple entries"""
        records = [{
            'profile': 'Ashwin',
            'date': '2026-01-06',
            'food_description': f'meal {i}',
            **sample_nutrition_data
        } for i in range(3)]
        df = pd.DataFrame(records)

        summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')

        assert summary['calories'] == 450 * 3
        assert summary['protein'] == 25.5 * 3
        assert summary['entries'] == 3

    def test_calculate_daily_summary_different_profiles(self, sample_nutrition_data):
        """Test daily summary filters by profile correctly"""
        records = [
            {'profile': 'Ashwin', 'date': '2026-01-06', 'food_description': 'eggs', **sample_nutrition_data},
            {'profile': 'Nandhitha', 'date': '2026-01-06', 'food_description': 'toast',
             'calories': 300, 'protein': 10.0, 'carbs': 40.0, 'fat': 8.0, 'sugar': 3.0, 'fiber': 2.0},
        ]
        df = pd.DataFrame(records)

        ashwin_summary = calculate_daily_summary(df, 'Ashwin', '2026-01-06')
        nandhitha_summary = calculate_daily_summary(df, 'Nandhitha', '2026-01-06')

        assert ashwin_summary['calories'] == 450
        assert ashwin_summary['entries'] == 1
        assert nandhitha_summary['calories'] == 300
        assert nandhitha_summary['entries'] == 1

    def test_calculate_daily_summary_different_dates(self, sample_nutrition_data):
        """Test daily summary filters by date correctly"""
        records = [
            {'profile': 'Ashwin', 'date': '2026-01-06', 'food_description': 'eggs', **sample_nutrition_data},
            {'profile': 'Ashwin', 'date': '2026-01-07', 'food_description': 'toast',
             'calories': 300, 'protein': 10.0, 'carbs': 40.0, 'fat': 8.0, 'sugar': 3.0, 'fiber': 2.0},
        ]
        df = pd.DataFrame(records)

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

    def test_calculate_daily_summary_wrong_profile(self, sample_nutrition_data):
        """Test daily summary with non-matching profile"""
        df = pd.DataFrame([{
            'profile': 'Ashwin',
            'date': '2026-01-06',
            'food_description': 'eggs',
            **sample_nutrition_data
        }])

        summary = calculate_daily_summary(df, 'NonExistentProfile', '2026-01-06')

        assert summary['entries'] == 0
        assert summary['calories'] == 0
