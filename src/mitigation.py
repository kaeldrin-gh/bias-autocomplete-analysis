"""
Bias Mitigation Strategies Module

This module provides algorithmic approaches and prototypes for mitigating bias
in autocomplete suggestions through fairness-aware re-ranking and filtering.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import math
from sklearn.preprocessing import MinMaxScaler


class FairnessCalculator:
    """Calculates fairness scores for autocomplete suggestions."""
    
    def __init__(self) -> None:
        """Initialize the fairness calculator."""
        self.logger = logging.getLogger(__name__)
        
        # Define fairness criteria weights
        self.fairness_criteria = {
            'demographic_balance': 0.3,
            'sentiment_neutrality': 0.3,
            'diversity_bonus': 0.2,
            'stereotype_penalty': 0.2
        }
        
        # Stereotype patterns to penalize
        self.negative_stereotypes = {
            'gender_stereotypes': [
                'women in kitchen', 'women in fashion', 'men in construction',
                'women in nursing', 'men in engineering'
            ],
            'profession_stereotypes': [
                'people of color in sports', 'women in administrative',
                'men in leadership'
            ]
        }
    
    def calculate_demographic_balance_score(self, suggestion: str, context: Dict[str, Any]) -> float:
        """
        Calculate demographic balance score for a suggestion.
        
        Args:
            suggestion: The autocomplete suggestion text
            context: Context information about the query and existing suggestions
            
        Returns:
            Demographic balance score (0-1, higher is more balanced)
        """
        # Check for demographic terms
        suggestion_lower = suggestion.lower()
        
        gender_terms = {
            'male': ['man', 'men', 'male', 'masculine', 'father', 'husband'],
            'female': ['woman', 'women', 'female', 'feminine', 'mother', 'wife']
        }
        
        # Count gender representation in current suggestion set
        existing_suggestions = context.get('existing_suggestions', [])
        
        male_count = sum(1 for s in existing_suggestions if any(term in s.lower() for term in gender_terms['male']))
        female_count = sum(1 for s in existing_suggestions if any(term in s.lower() for term in gender_terms['female']))
        
        # Determine if current suggestion adds balance
        has_male_terms = any(term in suggestion_lower for term in gender_terms['male'])
        has_female_terms = any(term in suggestion_lower for term in gender_terms['female'])
        
        if has_male_terms and male_count > female_count:
            return 0.3  # Reduces balance
        elif has_female_terms and female_count > male_count:
            return 0.3  # Reduces balance
        elif has_male_terms and male_count <= female_count:
            return 0.8  # Improves balance
        elif has_female_terms and female_count <= male_count:
            return 0.8  # Improves balance
        else:
            return 0.6  # Neutral
    
    def calculate_sentiment_neutrality_score(self, suggestion: str, sentiment_scores: Dict[str, float]) -> float:
        """
        Calculate sentiment neutrality score.
        
        Args:
            suggestion: The autocomplete suggestion text
            sentiment_scores: Sentiment analysis results
            
        Returns:
            Sentiment neutrality score (0-1, higher is more neutral/positive)
        """
        positive_score = sentiment_scores.get('POSITIVE', 0.0)
        negative_score = sentiment_scores.get('NEGATIVE', 0.0)
        neutral_score = sentiment_scores.get('NEUTRAL', 0.0)
        
        # Prefer positive and neutral sentiments, penalize negative
        if negative_score > 0.6:
            return 0.2  # Strong negative sentiment
        elif positive_score > 0.6:
            return 1.0  # Strong positive sentiment
        elif neutral_score > 0.5:
            return 0.8  # Neutral sentiment
        else:
            return 0.5  # Mixed sentiment
    
    def calculate_diversity_score(self, suggestion: str, context: Dict[str, Any]) -> float:
        """
        Calculate diversity bonus for unique or varied suggestions.
        
        Args:
            suggestion: The autocomplete suggestion text
            context: Context information including existing suggestions
            
        Returns:
            Diversity score (0-1, higher for more diverse suggestions)
        """
        existing_suggestions = context.get('existing_suggestions', [])
        
        if not existing_suggestions:
            return 1.0
        
        # Calculate semantic similarity with existing suggestions
        suggestion_words = set(suggestion.lower().split())
        
        max_overlap = 0
        for existing in existing_suggestions:
            existing_words = set(existing.lower().split())
            if len(existing_words) > 0:
                overlap = len(suggestion_words.intersection(existing_words))
                overlap_ratio = overlap / max(len(suggestion_words), len(existing_words))
                max_overlap = max(max_overlap, overlap_ratio)
        
        # Higher score for lower overlap (more diversity)
        return 1.0 - max_overlap
    
    def calculate_stereotype_penalty(self, suggestion: str) -> float:
        """
        Calculate penalty for stereotypical suggestions.
        
        Args:
            suggestion: The autocomplete suggestion text
            
        Returns:
            Stereotype penalty score (0-1, higher means less stereotypical)
        """
        suggestion_lower = suggestion.lower()
        
        # Check against known stereotype patterns
        stereotype_count = 0
        total_patterns = 0
        
        for category, patterns in self.negative_stereotypes.items():
            total_patterns += len(patterns)
            for pattern in patterns:
                if pattern in suggestion_lower:
                    stereotype_count += 1
        
        if total_patterns == 0:
            return 1.0
        
        # Higher score means fewer stereotypes
        stereotype_ratio = stereotype_count / total_patterns
        return 1.0 - stereotype_ratio
    
    def calculate_overall_fairness_score(
        self, 
        suggestion: str, 
        sentiment_scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive fairness score for a suggestion.
        
        Args:
            suggestion: The autocomplete suggestion text
            sentiment_scores: Sentiment analysis results
            context: Context information
            
        Returns:
            Dictionary containing individual and overall fairness scores
        """
        # Calculate individual components
        demo_balance = self.calculate_demographic_balance_score(suggestion, context)
        sentiment_neutrality = self.calculate_sentiment_neutrality_score(suggestion, sentiment_scores)
        diversity = self.calculate_diversity_score(suggestion, context)
        stereotype_penalty = self.calculate_stereotype_penalty(suggestion)
        
        # Calculate weighted overall score
        overall_score = (
            demo_balance * self.fairness_criteria['demographic_balance'] +
            sentiment_neutrality * self.fairness_criteria['sentiment_neutrality'] +
            diversity * self.fairness_criteria['diversity_bonus'] +
            stereotype_penalty * self.fairness_criteria['stereotype_penalty']
        )
        
        return {
            'demographic_balance': demo_balance,
            'sentiment_neutrality': sentiment_neutrality,
            'diversity': diversity,
            'stereotype_penalty': stereotype_penalty,
            'overall_fairness': overall_score
        }


class RelevanceCalculator:
    """Calculates relevance scores for autocomplete suggestions."""
    
    def __init__(self) -> None:
        """Initialize the relevance calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_string_similarity(self, query: str, suggestion: str) -> float:
        """
        Calculate string-based similarity between query and suggestion.
        
        Args:
            query: User's input query
            suggestion: Autocomplete suggestion
            
        Returns:
            Similarity score (0-1)
        """
        query_words = set(query.lower().split())
        suggestion_words = set(suggestion.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate word overlap
        intersection = query_words.intersection(suggestion_words)
        union = query_words.union(suggestion_words)
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Bonus for prefix matching
        prefix_bonus = 0.0
        if suggestion.lower().startswith(query.lower()):
            prefix_bonus = 0.3
        
        return min(1.0, jaccard_similarity + prefix_bonus)
    
    def calculate_popularity_score(self, suggestion: str, popularity_data: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate popularity-based relevance score.
        
        Args:
            suggestion: Autocomplete suggestion
            popularity_data: Optional popularity metrics
            
        Returns:
            Popularity score (0-1)
        """
        if popularity_data and suggestion in popularity_data:
            return popularity_data[suggestion]
        
        # Fallback: simple heuristic based on suggestion length and common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        suggestion_words = suggestion.lower().split()
        
        # Shorter suggestions with fewer common words are often more specific/relevant
        length_penalty = min(1.0, 10 / max(1, len(suggestion)))
        common_word_penalty = 1.0 - (sum(1 for word in suggestion_words if word in common_words) / max(1, len(suggestion_words)))
        
        return (length_penalty + common_word_penalty) / 2
    
    def calculate_overall_relevance_score(
        self, 
        query: str, 
        suggestion: str,
        popularity_data: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overall relevance score combining multiple factors.
        
        Args:
            query: User's input query
            suggestion: Autocomplete suggestion
            popularity_data: Optional popularity metrics
            
        Returns:
            Overall relevance score (0-1)
        """
        similarity_score = self.calculate_string_similarity(query, suggestion)
        popularity_score = self.calculate_popularity_score(suggestion, popularity_data)
        
        # Weighted combination
        relevance_score = 0.7 * similarity_score + 0.3 * popularity_score
        
        return relevance_score


class FairnessAwareReranker:
    """Implements fairness-aware re-ranking of autocomplete suggestions."""
    
    def __init__(self, fairness_weight: float = 0.3) -> None:
        """
        Initialize the fairness-aware reranker.
        
        Args:
            fairness_weight: Weight for fairness vs relevance (0-1)
        """
        self.fairness_weight = fairness_weight
        self.relevance_weight = 1.0 - fairness_weight
        
        self.fairness_calculator = FairnessCalculator()
        self.relevance_calculator = RelevanceCalculator()
        self.logger = logging.getLogger(__name__)
    
    def rerank_suggestions(
        self, 
        suggestions: List[str], 
        query: str,
        sentiment_scores_list: Optional[List[Dict[str, float]]] = None,
        popularity_data: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Re-rank suggestions based on combined fairness and relevance scores.
        
        Args:
            suggestions: List of autocomplete suggestions
            query: User's input query
            sentiment_scores_list: Optional list of sentiment scores for each suggestion
            popularity_data: Optional popularity data
            
        Returns:
            List of (suggestion, combined_score, score_breakdown) tuples, sorted by score
        """
        self.logger.info(f"Re-ranking {len(suggestions)} suggestions with fairness_weight={self.fairness_weight}")
        
        if not suggestions:
            return []
        
        # Prepare sentiment scores
        if sentiment_scores_list is None:
            sentiment_scores_list = [{'POSITIVE': 0.5, 'NEGATIVE': 0.3, 'NEUTRAL': 0.2} for _ in suggestions]
        
        scored_suggestions = []
        
        for i, suggestion in enumerate(suggestions):
            # Calculate relevance score
            relevance_score = self.relevance_calculator.calculate_overall_relevance_score(
                query, suggestion, popularity_data
            )
            
            # Calculate fairness score
            context = {
                'existing_suggestions': suggestions[:i],  # Suggestions processed so far
                'query': query
            }
            
            sentiment_scores = sentiment_scores_list[i] if i < len(sentiment_scores_list) else {}
            fairness_scores = self.fairness_calculator.calculate_overall_fairness_score(
                suggestion, sentiment_scores, context
            )
            
            # Combine scores
            combined_score = (
                self.relevance_weight * relevance_score +
                self.fairness_weight * fairness_scores['overall_fairness']
            )
            
            # Prepare score breakdown
            score_breakdown = {
                'relevance_score': relevance_score,
                'fairness_score': fairness_scores['overall_fairness'],
                'combined_score': combined_score,
                'fairness_components': fairness_scores
            }
            
            scored_suggestions.append((suggestion, combined_score, score_breakdown))
        
        # Sort by combined score (descending)
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Re-ranking completed. Top suggestion: '{scored_suggestions[0][0]}' with score {scored_suggestions[0][1]:.3f}")
        
        return scored_suggestions


class BiasFilter:
    """Filters out biased or problematic suggestions."""
    
    def __init__(self, strictness: float = 0.5) -> None:
        """
        Initialize the bias filter.
        
        Args:
            strictness: Filtering strictness (0-1, higher is more strict)
        """
        self.strictness = strictness
        self.logger = logging.getLogger(__name__)
        
        # Define filtering rules
        self.banned_patterns = [
            r'\bwomen in kitchen\b',
            r'\bmen in construction\b',
            r'\bpeople of color in sports\b'
        ]
        
        self.warning_patterns = [
            r'\bstereotype\b',
            r'\bgender role\b',
            r'\btraditional\b'
        ]
    
    def should_filter_suggestion(
        self, 
        suggestion: str, 
        fairness_scores: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Determine if a suggestion should be filtered out.
        
        Args:
            suggestion: The autocomplete suggestion
            fairness_scores: Fairness scores for the suggestion
            
        Returns:
            Tuple of (should_filter, reason)
        """
        import re
        
        suggestion_lower = suggestion.lower()
        
        # Check banned patterns
        for pattern in self.banned_patterns:
            if re.search(pattern, suggestion_lower):
                return True, f"Contains banned pattern: {pattern}"
        
        # Check fairness thresholds
        overall_fairness = fairness_scores.get('overall_fairness', 0.5)
        stereotype_penalty = fairness_scores.get('stereotype_penalty', 0.5)
        
        fairness_threshold = 0.3 + (self.strictness * 0.4)  # 0.3-0.7 range
        
        if overall_fairness < fairness_threshold:
            return True, f"Overall fairness score {overall_fairness:.3f} below threshold {fairness_threshold:.3f}"
        
        if stereotype_penalty < 0.2:
            return True, f"High stereotype content detected (score: {stereotype_penalty:.3f})"
        
        return False, ""
    
    def filter_suggestions(
        self, 
        scored_suggestions: List[Tuple[str, float, Dict[str, float]]]
    ) -> Tuple[List[Tuple[str, float, Dict[str, float]]], List[Tuple[str, str]]]:
        """
        Filter suggestions based on bias criteria.
        
        Args:
            scored_suggestions: List of (suggestion, score, score_breakdown) tuples
            
        Returns:
            Tuple of (filtered_suggestions, rejected_suggestions_with_reasons)
        """
        filtered_suggestions = []
        rejected_suggestions = []
        
        for suggestion, score, score_breakdown in scored_suggestions:
            fairness_scores = score_breakdown.get('fairness_components', {})
            
            should_filter, reason = self.should_filter_suggestion(suggestion, fairness_scores)
            
            if should_filter:
                rejected_suggestions.append((suggestion, reason))
                self.logger.info(f"Filtered suggestion '{suggestion}': {reason}")
            else:
                filtered_suggestions.append((suggestion, score, score_breakdown))
        
        self.logger.info(f"Filtered {len(rejected_suggestions)} suggestions, kept {len(filtered_suggestions)}")
        
        return filtered_suggestions, rejected_suggestions


class MitigationPipeline:
    """Complete bias mitigation pipeline combining re-ranking and filtering."""
    
    def __init__(self, fairness_weight: float = 0.3, filter_strictness: float = 0.5) -> None:
        """
        Initialize the complete mitigation pipeline.
        
        Args:
            fairness_weight: Weight for fairness in re-ranking (0-1)
            filter_strictness: Strictness of bias filtering (0-1)
        """
        self.fairness_weight = fairness_weight
        self.filter_strictness = filter_strictness
        
        self.reranker = FairnessAwareReranker(fairness_weight)
        self.filter = BiasFilter(filter_strictness)
        self.logger = logging.getLogger(__name__)
    
    def mitigate_bias(
        self, 
        suggestions: List[str], 
        query: str,
        sentiment_scores_list: Optional[List[Dict[str, float]]] = None,
        max_suggestions: int = 10
    ) -> Dict[str, Any]:
        """
        Apply complete bias mitigation pipeline to suggestions.
        
        Args:
            suggestions: Original list of suggestions
            query: User's input query
            sentiment_scores_list: Optional sentiment scores
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            Dictionary containing mitigated suggestions and pipeline results
        """
        self.logger.info(f"Starting bias mitigation pipeline for query: '{query}'")
        
        # Step 1: Re-rank suggestions
        scored_suggestions = self.reranker.rerank_suggestions(
            suggestions, query, sentiment_scores_list
        )
        
        # Step 2: Filter biased suggestions
        filtered_suggestions, rejected_suggestions = self.filter.filter_suggestions(scored_suggestions)
        
        # Step 3: Limit to max suggestions
        final_suggestions = filtered_suggestions[:max_suggestions]
        
        # Prepare results
        results = {
            'original_suggestions': suggestions,
            'mitigated_suggestions': [s[0] for s in final_suggestions],
            'suggestion_scores': [
                {
                    'suggestion': s[0],
                    'combined_score': s[1],
                    'score_breakdown': s[2]
                } for s in final_suggestions
            ],
            'rejected_suggestions': rejected_suggestions,
            'pipeline_stats': {
                'original_count': len(suggestions),
                'after_reranking': len(scored_suggestions),
                'after_filtering': len(filtered_suggestions),
                'final_count': len(final_suggestions),
                'rejection_rate': len(rejected_suggestions) / max(1, len(suggestions))
            },
            'configuration': {
                'fairness_weight': self.fairness_weight,
                'filter_strictness': self.filter_strictness
            }
        }
        
        self.logger.info(f"Mitigation completed: {results['pipeline_stats']['final_count']} final suggestions")
        
        return results


# Algorithmic Strategy Documentation
MITIGATION_STRATEGIES = {
    'fairness_aware_reranking': {
        'description': 'Re-ranks suggestions based on combined fairness and relevance scores',
        'algorithm': 'combined_score = (1-α) * relevance_score + α * fairness_score',
        'parameters': ['fairness_weight (α)'],
        'strengths': ['Balances fairness and utility', 'Transparent scoring', 'Configurable trade-offs'],
        'limitations': ['Requires good fairness metrics', 'May reduce relevance', 'Computational overhead'],
        'deployment_considerations': [
            'A/B testing for fairness_weight optimization',
            'Real-time performance monitoring',
            'User feedback integration',
            'Regular model updates'
        ]
    },
    'bias_filtering': {
        'description': 'Filters out suggestions that fail fairness criteria',
        'algorithm': 'filter(suggestion) = fairness_score > threshold AND not in banned_patterns',
        'parameters': ['strictness_threshold', 'banned_patterns'],
        'strengths': ['Hard constraints on bias', 'Interpretable rules', 'Fast execution'],
        'limitations': ['May over-filter', 'Rule-based limitations', 'Requires pattern updates'],
        'deployment_considerations': [
            'Regular pattern updates',
            'Monitoring filter effectiveness',
            'Balancing coverage and precision',
            'User impact assessment'
        ]
    },
    'demographic_balancing': {
        'description': 'Ensures balanced representation across demographic groups',
        'algorithm': 'Quota-based selection ensuring proportional representation',
        'parameters': ['target_proportions', 'minimum_thresholds'],
        'strengths': ['Guaranteed representation', 'Clear fairness metrics', 'Auditable results'],
        'limitations': ['May sacrifice relevance', 'Requires demographic classification', 'Rigid constraints'],
        'deployment_considerations': [
            'Legal compliance verification',
            'Demographic classification accuracy',
            'User acceptance testing',
            'Performance impact assessment'
        ]
    }
}
