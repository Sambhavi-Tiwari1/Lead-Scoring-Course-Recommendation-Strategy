"""
Course recommendation engine using multiple approaches
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CourseRecommender:
    """
    Hybrid course recommendation system
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        self.similarity_metric = similarity_metric
        self.course_features = None
        self.course_matrix = None
        self.courses_df = None
        self.user_course_matrix = None
    
    def fit_courses(self, courses_df: pd.DataFrame, 
                    interactions_df: pd.DataFrame = None):
        """
        Fit course recommender with course data
        """
        self.courses_df = courses_df
        
        # Build course feature matrix for content-based filtering
        self._build_content_features(courses_df)
        
        # Build user-course interaction matrix for collaborative filtering
        if interactions_df is not None:
            self._build_collaborative_features(interactions_df)
        
        logger.info("Course recommender fitted successfully")
    
    def _build_content_features(self, courses_df: pd.DataFrame):
        """
        Build content-based features from course attributes
        """
        # Combine categorical features
        courses_df['combined_features'] = (
            courses_df['category'] + " " + 
            courses_df['subcategory'] + " " + 
            courses_df['difficulty'] + " " +
            courses_df['name']
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.course_matrix = tfidf.fit_transform(courses_df['combined_features'])
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_features = ['rating', 'num_reviews', 'price', 'popularity_score']
        numerical_matrix = scaler.fit_transform(courses_df[numerical_features])
        
        # Combine with TF-IDF (weighted)
        self.course_features = np.hstack([
            self.course_matrix.toarray(),
            numerical_matrix * 0.3  # Reduce weight of numerical features
        ])
        
        logger.info(f"Built content features for {len(courses_df)} courses")
    
    def _build_collaborative_features(self, interactions_df: pd.DataFrame):
        """
        Build collaborative filtering matrix
        """
        # Pivot table: customers x courses
        self.user_course_matrix = interactions_df.pivot_table(
            index='customer_id',
            columns='course_id',
            values='rating_given',
            aggfunc='mean'
        ).fillna(0)
        
        logger.info(f"Built collaborative matrix: {self.user_course_matrix.shape}")
    
    def recommend_content_based(self, customer_id: str, 
                               customer_views: List[str],
                               top_n: int = 5) -> List[Dict]:
        """
        Content-based recommendations based on customer browsing history
        """
        if customer_views and self.course_matrix is not None:
            # Get viewed course IDs
            viewed_courses = self.courses_df[
                self.courses_df['course_id'].isin(customer_views)
            ]
            
            if not viewed_courses.empty:
                # Average features of viewed courses
                viewed_indices = viewed_courses.index.tolist()
                viewed_features = self.course_features[viewed_indices].mean(axis=0)
                
                # Calculate similarity with all courses
                similarities = cosine_similarity(
                    viewed_features.reshape(1, -1),
                    self.course_features
                ).flatten()
                
                # Get top recommendations (excluding viewed)
                viewed_set = set(customer_views)
                recommendations = []
                
                for idx in np.argsort(similarities)[::-1]:
                    course_id = self.courses_df.iloc[idx]['course_id']
                    if course_id not in viewed_set:
                        recommendations.append({
                            'course_id': course_id,
                            'name': self.courses_df.iloc[idx]['name'],
                            'category': self.courses_df.iloc[idx]['category'],
                            'rating': self.courses_df.iloc[idx]['rating'],
                            'price': self.courses_df.iloc[idx]['price'],
                            'similarity_score': similarities[idx]
                        })
                        
                        if len(recommendations) >= top_n:
                            break
                
                return recommendations
        
        # Fallback: recommend top-rated courses
        return self.get_top_courses(top_n)
    
    def recommend_collaborative(self, customer_id: str, 
                               top_n: int = 5) -> List[Dict]:
        """
        Collaborative filtering recommendations based on similar users
        """
        if self.user_course_matrix is not None:
            if customer_id in self.user_course_matrix.index:
                # Find similar users
                user_vector = self.user_course_matrix.loc[customer_id].values.reshape(1, -1)
                similarities = cosine_similarity(
                    user_vector,
                    self.user_course_matrix.values
                ).flatten()
                
                # Get top similar users (excluding self)
                similar_users = np.argsort(similarities)[::-1][1:10]
                
                # Aggregate courses from similar users
                course_scores = {}
                for user_idx in similar_users:
                    user_courses = self.user_course_matrix.iloc[user_idx]
                    rated_courses = user_courses[user_courses > 0].index.tolist()
                    
                    for course in rated_courses:
                        if course not in course_scores:
                            course_scores[course] = 0
                        course_scores[course] += similarities[user_idx]
                
                # Get top recommendations
                sorted_courses = sorted(course_scores.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                # Get course details
                recommendations = []
                for course_id, score in sorted_courses[:top_n]:
                    course = self.courses_df[
                        self.courses_df['course_id'] == course_id
                    ]
                    if not course.empty:
                        recommendations.append({
                            'course_id': course_id,
                            'name': course.iloc[0]['name'],
                            'category': course.iloc[0]['category'],
                            'rating': course.iloc[0]['rating'],
                            'price': course.iloc[0]['price'],
                            'collaborative_score': score
                        })
                
                return recommendations
        
        # Fallback: top courses
        return self.get_top_courses(top_n)
    
    def recommend_hybrid(self, customer_id: str, 
                        customer_views: List[str],
                        top_n: int = 5) -> List[Dict]:
        """
        Hybrid recommendations combining content-based and collaborative
        """
        # Get recommendations from both methods
        content_recs = self.recommend_content_based(customer_id, customer_views, top_n*2)
        collab_recs = self.recommend_collaborative(customer_id, top_n*2)
        
        # Combine and deduplicate
        combined = {}
        
        for rec in content_recs:
            course_id = rec['course_id']
            if course_id not in combined:
                combined[course_id] = {
                    'content_score': rec.get('similarity_score', 0),
                    'collab_score': 0,
                    **rec
                }
            else:
                combined[course_id]['content_score'] = rec.get('similarity_score', 0)
        
        for rec in collab_recs:
            course_id = rec['course_id']
            if course_id not in combined:
                combined[course_id] = {
                    'content_score': 0,
                    'collab_score': rec.get('collaborative_score', 0),
                    **rec
                }
            else:
                combined[course_id]['collab_score'] = rec.get('collaborative_score', 0)
        
        # Calculate hybrid score (weighted average)
        for course_id in combined:
            content_score = combined[course_id]['content_score']
            collab_score = combined[course_id]['collab_score']
            
            # Normalize scores if needed
            combined[course_id]['hybrid_score'] = (
                0.6 * content_score + 0.4 * collab_score
            )
        
        # Sort by hybrid score
        sorted_recs = sorted(combined.values(),
                           key=lambda x: x.get('hybrid_score', 0),
                           reverse=True)
        
        return sorted_recs[:top_n]
    
    def get_top_courses(self, top_n: int = 5) -> List[Dict]:
        """
        Get top courses based on popularity
        """
        top_courses = self.courses_df.nlargest(top_n, 'popularity_score')
        
        recommendations = []
        for _, course in top_courses.iterrows():
            recommendations.append({
                'course_id': course['course_id'],
                'name': course['name'],
                'category': course['category'],
                'rating': course['rating'],
                'price': course['price'],
                'popularity_score': course['popularity_score']
            })
        
        return recommendations
    
    def get_recommendations(self, customer_id: str = None,
                           customer_data: Dict = None,
                           top_n: int = 5) -> List[Dict]:
        """
        Main recommendation method
        """
        if customer_data is None:
            # Fallback: top courses
            return self.get_top_courses(top_n)
        
        # Get customer views history
        customer_views = customer_data.get('course_views_history', [])
        
        # Use hybrid recommendation
        return self.recommend_hybrid(customer_id, customer_views, top_n)
