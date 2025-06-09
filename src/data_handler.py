"""
Data Generation and Processing Module

This module provides functionality to generate simulated autocomplete data
and process it for bias analysis. All data is synthetically generated to
avoid scraping live endpoints.
"""

import logging
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime


class AutocompleteDataGenerator:
    """Generates simulated autocomplete data for bias analysis."""
    
    def __init__(self, random_seed: int = 42) -> None:
        """
        Initialize the data generator with a random seed for reproducibility.
        
        Args:
            random_seed: Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define base categories and bias patterns
        self._initialize_data_templates()
    
    def _initialize_data_templates(self) -> None:
        """Initialize the data templates for generating biased and unbiased suggestions."""
        
        # Base profession queries
        self.professions = [
            "engineer", "doctor", "nurse", "teacher", "lawyer", "pilot", 
            "chef", "artist", "scientist", "manager", "developer", "designer"
        ]
        
        # Demographic-related queries
        self.demographic_queries = [
            "women in", "men in", "people of color in", "young people in",
            "older adults in", "students interested in"
        ]
        
        # Biased suggestion patterns (simulated negative stereotypes)
        self.biased_patterns = {
            "women in": [
                "women in the kitchen", "women in fashion", "women in nursing",
                "women in teaching", "women in administrative roles"
            ],
            "men in": [
                "men in engineering", "men in leadership", "men in technology",
                "men in finance", "men in construction"
            ],
            "people of color in": [
                "people of color in sports", "people of color in entertainment",
                "people of color in service industry"
            ]
        }
        
        # Neutral/positive suggestions
        self.positive_patterns = {
            "women in": [
                "women in STEM", "women in leadership", "women in technology",
                "women in engineering", "women in medicine"
            ],
            "men in": [
                "men in healthcare", "men in education", "men in social work",
                "men in psychology", "men in arts"
            ],
            "people of color in": [
                "people of color in academia", "people of color in technology",
                "people of color in leadership", "people of color in research"
            ]
        }
    
    def generate_base_queries(self, num_queries: int = 100) -> List[str]:
        """
        Generate a list of base search queries.
        
        Args:
            num_queries: Number of base queries to generate
            
        Returns:
            List of base query strings
        """
        self.logger.info(f"Generating {num_queries} base queries with seed {self.random_seed}")
        
        queries = []
        
        # Generate profession-based queries
        for _ in range(num_queries // 2):
            profession = random.choice(self.professions)
            queries.append(profession)
        
        # Generate demographic + profession queries
        for _ in range(num_queries // 2):
            demographic = random.choice(self.demographic_queries)
            profession = random.choice(self.professions)
            queries.append(f"{demographic} {profession}")
        
        self.logger.info(f"Generated {len(queries)} base queries")
        return queries
    
    def generate_autocomplete_suggestions(
        self, 
        base_query: str, 
        num_suggestions: int = 10,
        bias_probability: float = 0.3
    ) -> List[str]:
        """
        Generate autocomplete suggestions for a given base query.
        
        Args:
            base_query: The base search query
            num_suggestions: Number of suggestions to generate
            bias_probability: Probability of generating biased suggestions
            
        Returns:
            List of autocomplete suggestion strings
        """
        suggestions = []
        
        # Check if query matches demographic patterns
        matching_pattern = None
        for pattern in self.demographic_queries:
            if pattern in base_query.lower():
                matching_pattern = pattern
                break
        
        if matching_pattern and random.random() < bias_probability:
            # Generate biased suggestions
            if matching_pattern in self.biased_patterns:
                biased_options = self.biased_patterns[matching_pattern]
                suggestions.extend(random.sample(
                    biased_options, 
                    min(num_suggestions // 2, len(biased_options))
                ))
        
        # Fill remaining suggestions with neutral/positive options
        if matching_pattern and matching_pattern in self.positive_patterns:
            positive_options = self.positive_patterns[matching_pattern]
            remaining_slots = num_suggestions - len(suggestions)
            suggestions.extend(random.sample(
                positive_options,
                min(remaining_slots, len(positive_options))
            ))
        
        # Fill any remaining slots with generic completions
        while len(suggestions) < num_suggestions:
            profession = random.choice(self.professions)
            generic_suggestion = f"{base_query} {profession}"
            if generic_suggestion not in suggestions:
                suggestions.append(generic_suggestion)
        
        return suggestions[:num_suggestions]
    
    def generate_full_dataset(
        self, 
        num_base_queries: int = 100,
        suggestions_per_query: int = 10,
        bias_probability: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate a complete dataset of queries and autocomplete suggestions.
        
        Args:
            num_base_queries: Number of base queries to generate
            suggestions_per_query: Number of suggestions per query
            bias_probability: Probability of generating biased suggestions
            
        Returns:
            DataFrame containing queries, suggestions, and metadata
        """
        self.logger.info(
            f"Generating full dataset with {num_base_queries} queries, "
            f"{suggestions_per_query} suggestions each, bias_prob={bias_probability}"
        )
        
        # Generate base queries
        base_queries = self.generate_base_queries(num_base_queries)
        
        # Generate suggestions for each query
        dataset_rows = []
        for query_id, base_query in enumerate(base_queries):
            suggestions = self.generate_autocomplete_suggestions(
                base_query, suggestions_per_query, bias_probability
            )
            
            for suggestion_id, suggestion in enumerate(suggestions):
                dataset_rows.append({
                    'query_id': query_id,
                    'base_query': base_query,
                    'suggestion_id': suggestion_id,
                    'suggestion': suggestion,
                    'generation_timestamp': datetime.now().isoformat(),
                    'bias_probability': bias_probability,
                    'random_seed': self.random_seed
                })
        
        df = pd.DataFrame(dataset_rows)
        self.logger.info(f"Generated dataset with {len(df)} total suggestion records")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save the generated dataset to a CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the CSV file
        """
        df.to_csv(filepath, index=False)
        self.logger.info(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a previously generated dataset from CSV.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded dataset
    """
    logging.info(f"Loading dataset from {filepath}")
    return pd.read_csv(filepath)


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate a summary of the dataset for analysis.
    
    Args:
        df: DataFrame containing the dataset
        
    Returns:
        Dictionary containing dataset summary statistics
    """
    summary = {
        'total_suggestions': len(df),
        'unique_base_queries': df['base_query'].nunique(),
        'avg_suggestions_per_query': len(df) / df['base_query'].nunique(),
        'query_categories': df['base_query'].value_counts().to_dict(),
        'generation_parameters': {
            'bias_probability': df['bias_probability'].iloc[0] if not df.empty else None,
            'random_seed': df['random_seed'].iloc[0] if not df.empty else None
        }
    }
    
    logging.info(f"Dataset summary generated: {summary['total_suggestions']} suggestions")
    return summary
