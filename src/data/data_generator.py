"""
Synthetic data generator for lead scoring and course recommendation
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Generate synthetic customer, course, and interaction data
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
        self.locations = ["US", "UK", "Canada", "Australia", "India", "Germany"]
        self.income_levels = ["low", "medium", "high"]
        self.education_levels = ["high_school", "bachelors", "masters", "phd"]
        self.professions = ["student", "professional", "entrepreneur", "retired"]
        
        self.course_categories = {
            "Programming": ["Python", "Java", "JavaScript", "C++", "SQL"],
            "Data Science": ["Data Analysis", "Machine Learning", "Deep Learning", "Statistics"],
            "Business": ["Marketing", "Finance", "Management", "Entrepreneurship"],
            "Design": ["UI/UX", "Graphic Design", "Web Design", "Animation"]
        }
    
    def generate_customers(self, n_customers: int = 5000) -> pd.DataFrame:
        """Generate synthetic customer data"""
        customers = []
        
        for i in range(n_customers):
            age_group = random.choice(self.age_groups)
            location = random.choice(self.locations)
            income_level = random.choice(self.income_levels)
            education = random.choice(self.education_levels)
            profession = random.choice(self.professions)
            
            # Behavioral metrics
            website_visits = np.random.poisson(5) + 2
            time_spent = np.random.exponential(60) + 30
            pages_viewed = np.random.poisson(4) + 1
            course_views = np.random.poisson(2) + 1
            email_opens = np.random.poisson(3) + 1
            click_through_rate = np.random.beta(2, 5)
            
            # Engagement metrics
            social_media_interaction = np.random.poisson(2)
            content_shares = np.random.poisson(1)
            event_attendance = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Purchase history
            courses_purchased = np.random.poisson(1) + 1
            total_spent = courses_purchased * (np.random.uniform(50, 300))
            
            # Customer value calculation
            engagement_score = (website_visits * 0.2 + time_spent * 0.1 + 
                              pages_viewed * 0.15 + course_views * 0.2 + 
                              email_opens * 0.15 + click_through_rate * 0.2)
            
            customer = {
                "customer_id": f"CUST_{i+1:05d}",
                "age_group": age_group,
                "location": location,
                "income_level": income_level,
                "education": education,
                "profession": profession,
                "website_visits": website_visits,
                "time_spent": time_spent,
                "pages_viewed": pages_viewed,
                "course_views": course_views,
                "email_opens": email_opens,
                "click_through_rate": click_through_rate,
                "social_media_interaction": social_media_interaction,
                "content_shares": content_shares,
                "event_attendance": event_attendance,
                "courses_purchased": courses_purchased,
                "total_spent": total_spent,
                "engagement_score": engagement_score,
                "converted": int(engagement_score > 0.6 and random.random() < 0.4)
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        logger.info(f"Generated {len(df)} customers")
        return df
    
    def generate_courses(self, n_courses: int = 100) -> pd.DataFrame:
        """Generate synthetic course data"""
        courses = []
        
        for i in range(n_courses):
            category = random.choice(list(self.course_categories.keys()))
            subcategory = random.choice(self.course_categories[category])
            name = f"{subcategory} {random.choice(['101', 'Pro', 'Mastery', 'Advanced', 'Expert'])}"
            
            rating = np.random.beta(3, 1) * 4 + 1  # Scale from 1-5
            rating = min(5.0, max(1.0, rating))
            num_reviews = np.random.poisson(50) + 10
            
            # Social proof metrics
            has_social_proof = rating > 4.0 and num_reviews > 20
            
            course = {
                "course_id": f"COURSE_{i+1:05d}",
                "name": name,
                "category": category,
                "subcategory": subcategory,
                "rating": round(rating, 1),
                "num_reviews": num_reviews,
                "price": round(np.random.uniform(49, 399), 2),
                "duration_hours": round(np.random.uniform(5, 50), 1),
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "has_social_proof": has_social_proof,
                "popularity_score": rating * 0.3 + (num_reviews / 100) * 0.3 + 
                                  np.random.uniform(0, 0.4)
            }
            courses.append(course)
        
        df = pd.DataFrame(courses)
        logger.info(f"Generated {len(df)} courses")
        return df
    
    def generate_interactions(self, customers: pd.DataFrame, 
                             courses: pd.DataFrame,
                            n_interactions: int = 25000) -> pd.DataFrame:
        """Generate customer-course interactions"""
        interactions = []
        
        for i in range(n_interactions):
            customer = customers.sample(1).iloc[0]
            course = courses.sample(1).iloc[0]
            
            # Interaction type based on customer engagement
            engagement_score = customer['engagement_score']
            if engagement_score > 0.7:
                interaction_type = random.choices(['view', 'enroll', 'complete', 'review'],
                                                 weights=[0.3, 0.25, 0.25, 0.2])[0]
            elif engagement_score > 0.4:
                interaction_type = random.choices(['view', 'enroll', 'complete', 'review'],
                                                 weights=[0.5, 0.3, 0.15, 0.05])[0]
            else:
                interaction_type = random.choices(['view', 'enroll', 'complete', 'review'],
                                                 weights=[0.7, 0.2, 0.08, 0.02])[0]
            
            interaction = {
                "interaction_id": f"INT_{i+1:06d}",
                "customer_id": customer['customer_id'],
                "course_id": course['course_id'],
                "interaction_type": interaction_type,
                "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 180)),
                "rating_given": np.random.uniform(3, 5) if interaction_type == 'review' else None
            }
            interactions.append(interaction)
        
        df = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df)} interactions")
        return df
