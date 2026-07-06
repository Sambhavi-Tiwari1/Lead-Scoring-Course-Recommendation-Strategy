"""
Customer segmentation using multiple approaches
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CustomerSegmenter:
    """
    Segment customers based on behavioral and demographic data
    """
    
    def __init__(self, n_segments: int = 4):
        self.n_segments = n_segments
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_segments, random_state=42)
        self.segment_labels = None
        self.segment_profiles = {}
    
    def segment_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform customer segmentation using behavioral and demographic data
        """
        logger.info("Starting customer segmentation...")
        
        # Select features for segmentation
        behavioral_features = [
            'website_visits', 'time_spent', 'pages_viewed', 'course_views',
            'email_opens', 'click_through_rate', 'social_media_interaction',
            'content_shares', 'event_attendance', 'engagement_score'
        ]
        
        # Prepare feature matrix
        X = df[behavioral_features].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['pca_1'] = X_pca[:, 0]
        df['pca_2'] = X_pca[:, 1]
        
        # Perform K-means clustering
        self.kmeans.fit(X_scaled)
        df['segment'] = self.kmeans.labels_
        
        # Analyze segments
        self._analyze_segments(df)
        
        # Assign segment names
        self._assign_segment_names(df)
        
        logger.info(f"Segmentation complete. Found {self.n_segments} segments.")
        return df
    
    def _analyze_segments(self, df: pd.DataFrame):
        """Analyze and profile each segment"""
        for segment in range(self.n_segments):
            segment_data = df[df['segment'] == segment]
            
            profile = {
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                'avg_engagement': segment_data['engagement_score'].mean(),
                'avg_spent': segment_data['total_spent'].mean(),
                'avg_courses': segment_data['courses_purchased'].mean(),
                'conversion_rate': segment_data['converted'].mean() * 100,
                'demographics': {
                    'age_group': segment_data['age_group'].mode().iloc[0] if not segment_data.empty else None,
                    'income_level': segment_data['income_level'].mode().iloc[0] if not segment_data.empty else None,
                    'education': segment_data['education'].mode().iloc[0] if not segment_data.empty else None
                }
            }
            
            self.segment_profiles[segment] = profile
    
    def _assign_segment_names(self, df: pd.DataFrame):
        """Assign human-readable names to segments"""
        segment_mapping = {}
        
        for segment in range(self.n_segments):
            profile = self.segment_profiles[segment]
            engagement = profile['avg_engagement']
            conversion = profile['conversion_rate']
            spent = profile['avg_spent']
            
            if engagement > 0.7 and conversion > 30:
                name = "High-Value Customers"
            elif engagement > 0.5 and conversion > 20:
                name = "Engaged Learners"
            elif engagement > 0.3 and conversion > 10:
                name = "Casual Browsers"
            else:
                name = "Low-Engagement"
            
            segment_mapping[segment] = name
        
        df['segment_name'] = df['segment'].map(segment_mapping)
    
    def get_segment_recommendations(self, segment: str) -> Dict:
        """Get recommendations for a specific segment"""
        recommendations = {
            "High-Value Customers": {
                "strategy": "Premium Engagement",
                "actions": [
                    "Provide exclusive content and early access",
                    "Offer premium courses and certifications",
                    "Implement referral rewards program",
                    "Schedule personalized coaching sessions"
                ],
                "marketing_channels": ["Email", "Social Media", "Direct Mail"]
            },
            "Engaged Learners": {
                "strategy": "Retention & Upselling",
                "actions": [
                    "Send personalized course recommendations",
                    "Offer bundle discounts on related courses",
                    "Create community forums for engagement",
                    "Implement gamification and achievements"
                ],
                "marketing_channels": ["Email", "Social Media", "Content Marketing"]
            },
            "Casual Browsers": {
                "strategy": "Conversion Optimization",
                "actions": [
                    "Send targeted email campaigns with offers",
                    "Provide free introductory content",
                    "Show social proof and testimonials",
                    "Offer limited-time discounts"
                ],
                "marketing_channels": ["Search Ads", "Social Media", "Content Marketing"]
            },
            "Low-Engagement": {
                "strategy": "Reactivation",
                "actions": [
                    "Send re-engagement email campaigns",
                    "Offer free courses or trials",
                    "Survey to understand barriers",
                    "Provide personalized content recommendations"
                ],
                "marketing_channels": ["Email", "Search Ads", "Retargeting"]
            }
        }
        
        return recommendations.get(segment, {
            "strategy": "General Engagement",
            "actions": ["Continue engagement efforts"],
            "marketing_channels": ["All channels"]
        })
    
    def visualize_segments(self, df: pd.DataFrame, save_path: str = None):
        """Visualize customer segments"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Segment distribution
        ax = axes[0, 0]
        segment_counts = df['segment_name'].value_counts()
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        ax.set_title('Customer Segment Distribution')
        
        # Segment characteristics
        ax = axes[0, 1]
        segment_metrics = df.groupby('segment_name').agg({
            'engagement_score': 'mean',
            'courses_purchased': 'mean',
            'total_spent': 'mean'
        }).reset_index()
        
        segment_metrics_melted = pd.melt(segment_metrics, 
                                         id_vars=['segment_name'],
                                         value_vars=['engagement_score', 'courses_purchased', 'total_spent'],
                                         var_name='metric', value_name='value')
        
        sns.barplot(data=segment_metrics_melted, x='segment_name', y='value', hue='metric', ax=ax)
        ax.set_title('Segment Characteristics')
        ax.set_xlabel('Segment')
        ax.set_ylabel('Score')
        ax.legend()
        
        # PCA visualization
        ax = axes[1, 0]
        scatter = ax.scatter(df['pca_1'], df['pca_2'], c=df['segment'], cmap='viridis', alpha=0.6)
        ax.set_title('PCA Visualization of Segments')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        plt.colorbar(scatter, ax=ax)
        
        # Conversion rates by segment
        ax = axes[1, 1]
        conversion_rates = df.groupby('segment_name')['converted'].mean() * 100
        conversion_rates.sort_values().plot(kind='barh', ax=ax)
        ax.set_title('Conversion Rates by Segment')
        ax.set_xlabel('Conversion Rate (%)')
        ax.set_ylabel('Segment')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segment visualization saved to {save_path}")
        
        plt.show()
