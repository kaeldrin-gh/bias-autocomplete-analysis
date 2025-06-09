"""
NLP Analysis Pipeline Module

This module provides comprehensive NLP analysis capabilities for bias detection
in autocomplete suggestions, including sentiment analysis, semantic similarity,
and statistical bias quantification.
"""

import logging
import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
import nltk
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class TextProcessor:
    """Handles text preprocessing and normalization tasks."""
    
    def __init__(self) -> None:
        """Initialize the text processor with NLTK resources."""
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to lowercase and removing punctuation.
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text: Input text string
            
        Returns:
            List of extracted keywords
        """
        normalized = self.normalize_text(text)
        without_stopwords = self.remove_stopwords(normalized)
        keywords = without_stopwords.split()
        return [kw for kw in keywords if len(kw) > 2]


class SentimentAnalyzer:
    """Performs sentiment analysis on text using transformers."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained sentiment analysis model
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                return_all_scores=True
            )
            self.logger.info(f"Sentiment analyzer initialized with model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load model {model_name}, using default: {e}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                return_all_scores=True
            )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of input text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing sentiment scores
        """
        if not text or not isinstance(text, str):
            return {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}
        try:
            results = self.sentiment_pipeline(text)
            # Robust label mapping for different model outputs
            label_map = {
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL",
                "LABEL_2": "POSITIVE",
                "NEGATIVE": "NEGATIVE",
                "NEUTRAL": "NEUTRAL",
                "POSITIVE": "POSITIVE"
            }
            sentiment_scores = {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}
            for result in results[0]:  # First (and only) text result
                label = label_map.get(result['label'], result['label'].upper())
                if label in sentiment_scores:
                    sentiment_scores[label] = result['score']
            return sentiment_scores
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed for text '{text}': {e}")
            return {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}
    
    def get_dominant_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Get the dominant sentiment and its score.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (dominant_sentiment, confidence_score)
        """
        sentiment_scores = self.analyze_sentiment(text)
        if not sentiment_scores:
            return ("NEUTRAL", 0.0)
        
        dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[dominant_sentiment]
        
        return (dominant_sentiment, confidence)


class SemanticAnalyzer:
    """Performs semantic similarity analysis using transformer embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize semantic analyzer with sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Semantic analyzer initialized with model: {model_name}")
        except ImportError:
            self.logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self.model = None
            self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is not None:
            # Use sentence transformer
            embeddings = self.model.encode(texts)
            return embeddings
        else:
            # Fallback to TF-IDF
            self.logger.info("Using TF-IDF embeddings as fallback")
            embeddings = self.tfidf.fit_transform(texts)
            return embeddings.toarray()
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Similarity matrix as numpy array
        """
        embeddings = self.get_embeddings(texts)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def find_similar_suggestions(
        self, 
        target_text: str, 
        candidate_texts: List[str], 
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find suggestions similar to a target text.
        
        Args:
            target_text: Reference text
            candidate_texts: List of candidate texts to compare
            threshold: Minimum similarity threshold
            
        Returns:
            List of (text, similarity_score) tuples above threshold
        """
        all_texts = [target_text] + candidate_texts
        similarity_matrix = self.calculate_similarity_matrix(all_texts)
        
        target_similarities = similarity_matrix[0, 1:]  # First row, excluding self-similarity
        
        similar_suggestions = []
        for i, similarity in enumerate(target_similarities):
            if similarity >= threshold:
                similar_suggestions.append((candidate_texts[i], similarity))
        
        # Sort by similarity score (descending)
        similar_suggestions.sort(key=lambda x: x[1], reverse=True)
        return similar_suggestions


class BiasQuantifier:
    """Quantifies bias in autocomplete suggestions using statistical methods."""
    
    def __init__(self) -> None:
        """Initialize the bias quantifier."""
        self.logger = logging.getLogger(__name__)
        self.text_processor = TextProcessor()
        
        # Define bias-related keywords
        self.gender_keywords = {
            'male': ['man', 'men', 'male', 'masculine', 'father', 'husband', 'boy'],
            'female': ['woman', 'women', 'female', 'feminine', 'mother', 'wife', 'girl']
        }
        
        self.profession_stereotypes = {
            'technical': ['engineer', 'developer', 'programmer', 'scientist', 'analyst'],
            'care': ['nurse', 'teacher', 'social worker', 'counselor', 'therapist'],
            'leadership': ['ceo', 'manager', 'director', 'executive', 'leader']
        }
    
    def extract_demographic_indicators(self, text: str) -> Dict[str, bool]:
        """
        Extract demographic indicators from text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary indicating presence of demographic markers
        """
        normalized_text = self.text_processor.normalize_text(text)
        
        indicators = {
            'contains_male_terms': any(term in normalized_text for term in self.gender_keywords['male']),
            'contains_female_terms': any(term in normalized_text for term in self.gender_keywords['female']),
            'contains_technical_terms': any(term in normalized_text for term in self.profession_stereotypes['technical']),
            'contains_care_terms': any(term in normalized_text for term in self.profession_stereotypes['care']),
            'contains_leadership_terms': any(term in normalized_text for term in self.profession_stereotypes['leadership'])
        }
        
        return indicators
    
    def calculate_sentiment_bias(
        self, 
        df: pd.DataFrame, 
        group_column: str,
        sentiment_column: str
    ) -> Dict[str, Any]:
        """
        Calculate sentiment bias between different groups.
        
        Args:
            df: DataFrame containing suggestions and group labels
            group_column: Column name for grouping variable
            sentiment_column: Column name for sentiment scores
            
        Returns:
            Dictionary containing bias metrics and statistical tests
        """
        bias_results = {}
        
        # Get unique groups
        groups = df[group_column].unique()
        
        # Calculate sentiment statistics per group
        group_stats = {}
        group_sentiments = {}
        
        for group in groups:
            group_data = df[df[group_column] == group]
            sentiments = group_data[sentiment_column].dropna()
            
            if len(sentiments) > 0:
                group_stats[group] = {
                    'mean_sentiment': sentiments.mean(),
                    'std_sentiment': sentiments.std(),
                    'count': len(sentiments),
                    'positive_ratio': (sentiments > 0.5).mean(),
                    'negative_ratio': (sentiments < -0.5).mean()
                }
                group_sentiments[group] = sentiments.tolist()
        
        bias_results['group_statistics'] = group_stats
        
        # Perform statistical tests
        if len(groups) >= 2:
            sentiment_lists = [group_sentiments[group] for group in groups if group in group_sentiments]
            
            if len(sentiment_lists) >= 2 and all(len(lst) > 0 for lst in sentiment_lists):
                # ANOVA test for multiple groups
                if len(sentiment_lists) > 2:
                    f_stat, p_value = stats.f_oneway(*sentiment_lists)
                    bias_results['anova_test'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                
                # Pairwise t-tests
                pairwise_results = {}
                for i, group1 in enumerate(groups):
                    for j, group2 in enumerate(groups[i+1:], i+1):
                        if group1 in group_sentiments and group2 in group_sentiments:
                            t_stat, p_val = stats.ttest_ind(
                                group_sentiments[group1], 
                                group_sentiments[group2]
                            )
                            pairwise_results[f"{group1}_vs_{group2}"] = {
                                't_statistic': t_stat,
                                'p_value': p_val,
                                'significant': p_val < 0.05
                            }
                
                bias_results['pairwise_tests'] = pairwise_results
        
        self.logger.info(f"Calculated sentiment bias for {len(groups)} groups")
        return bias_results
    
    def calculate_representation_bias(self, df: pd.DataFrame, category_column: str) -> Dict[str, Any]:
        """
        Calculate representation bias in suggestion categories.
        
        Args:
            df: DataFrame containing suggestions
            category_column: Column name for categories
            
        Returns:
            Dictionary containing representation bias metrics
        """
        # Calculate category frequencies
        category_counts = df[category_column].value_counts()
        total_suggestions = len(df)
        
        # Calculate expected equal representation
        num_categories = len(category_counts)
        expected_proportion = 1.0 / num_categories
        
        # Chi-square test for equal representation
        expected_counts = [total_suggestions * expected_proportion] * num_categories
        observed_counts = category_counts.values
        
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        representation_results = {
            'category_counts': category_counts.to_dict(),
            'category_proportions': (category_counts / total_suggestions).to_dict(),
            'expected_proportion': expected_proportion,
            'chi_square_test': {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'representation_ratio': {
                cat: (count / total_suggestions) / expected_proportion 
                for cat, count in category_counts.items()
            }
        }
        
        self.logger.info(f"Calculated representation bias for {num_categories} categories")
        return representation_results


class AnalysisPipeline:
    """Orchestrates the complete NLP analysis pipeline."""
    
    def __init__(self) -> None:
        """Initialize the analysis pipeline with all components."""
        self.logger = logging.getLogger(__name__)
        
        self.text_processor = TextProcessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.bias_quantifier = BiasQuantifier()
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete analysis pipeline on a dataset.
        
        Args:
            df: DataFrame containing autocomplete suggestions
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting comprehensive analysis pipeline")
        
        results = {
            'preprocessing': {},
            'sentiment_analysis': {},
            'semantic_analysis': {},
            'bias_quantification': {}
        }
        
        # Text preprocessing
        self.logger.info("Performing text preprocessing")
        df['normalized_suggestion'] = df['suggestion'].apply(self.text_processor.normalize_text)
        df['suggestion_keywords'] = df['suggestion'].apply(self.text_processor.extract_keywords)
        
        results['preprocessing']['total_suggestions'] = len(df)
        results['preprocessing']['unique_suggestions'] = df['suggestion'].nunique()
        
        # Sentiment analysis
        self.logger.info("Performing sentiment analysis")
        sentiment_results = []
        for suggestion in df['suggestion']:
            sentiment_dict = self.sentiment_analyzer.analyze_sentiment(suggestion)
            dominant_sentiment, confidence = self.sentiment_analyzer.get_dominant_sentiment(suggestion)
            sentiment_results.append({
                'sentiment_dominant_sentiment': dominant_sentiment,
                'sentiment_confidence': confidence,
                'sentiment_POSITIVE': sentiment_dict.get('POSITIVE', 0.0),
                'sentiment_NEGATIVE': sentiment_dict.get('NEGATIVE', 0.0),
                'sentiment_NEUTRAL': sentiment_dict.get('NEUTRAL', 0.0)
            })
        sentiment_df = pd.DataFrame(sentiment_results)
        for col in sentiment_df.columns:
            df[col] = sentiment_df[col]
        
        results['sentiment_analysis']['sentiment_distribution'] = df['sentiment_dominant_sentiment'].value_counts().to_dict()
        results['sentiment_analysis']['average_confidence'] = df['sentiment_confidence'].mean()
        
        # Semantic analysis
        self.logger.info("Performing semantic similarity analysis")
        unique_suggestions = df['suggestion'].unique()
        
        if len(unique_suggestions) > 1:
            similarity_matrix = self.semantic_analyzer.calculate_similarity_matrix(unique_suggestions.tolist())
            
            # Calculate average similarity metrics
            upper_triangle = np.triu(similarity_matrix, k=1)
            non_zero_similarities = upper_triangle[upper_triangle > 0]
            
            results['semantic_analysis']['average_similarity'] = non_zero_similarities.mean()
            results['semantic_analysis']['max_similarity'] = non_zero_similarities.max()
            results['semantic_analysis']['similarity_std'] = non_zero_similarities.std()
        
        # Extract demographic indicators
        self.logger.info("Extracting demographic indicators")
        demographic_indicators = []
        for suggestion in df['suggestion']:
            indicators = self.bias_quantifier.extract_demographic_indicators(suggestion)
            demographic_indicators.append(indicators)
        
        demo_df = pd.DataFrame(demographic_indicators)
        for col in demo_df.columns:
            df[col] = demo_df[col]
        
        # Bias quantification
        self.logger.info("Quantifying bias metrics")
        
        # Create bias categories for analysis
        df['gender_category'] = 'neutral'
        df.loc[df['contains_male_terms'], 'gender_category'] = 'male'
        df.loc[df['contains_female_terms'], 'gender_category'] = 'female'
        
        df['profession_category'] = 'general'
        df.loc[df['contains_technical_terms'], 'profession_category'] = 'technical'
        df.loc[df['contains_care_terms'], 'profession_category'] = 'care'
        df.loc[df['contains_leadership_terms'], 'profession_category'] = 'leadership'
        
        # Sentiment bias analysis
        if 'sentiment_POSITIVE' in df.columns:
            sentiment_bias = self.bias_quantifier.calculate_sentiment_bias(
                df, 'gender_category', 'sentiment_POSITIVE'
            )
            results['bias_quantification']['sentiment_bias_by_gender'] = sentiment_bias
        
        # Representation bias analysis
        gender_bias = self.bias_quantifier.calculate_representation_bias(df, 'gender_category')
        profession_bias = self.bias_quantifier.calculate_representation_bias(df, 'profession_category')
        
        results['bias_quantification']['gender_representation_bias'] = gender_bias
        results['bias_quantification']['profession_representation_bias'] = profession_bias
        
        self.logger.info("Analysis pipeline completed successfully")
        
        return results, df
