#!/usr/bin/env python
"""
Main application for Lead Scoring & Course Recommendation
"""
import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_generator import DataGenerator
from src.segmentation.customer_segmenter import CustomerSegmenter
from src.scoring.lead_scorer import LeadScorer
from src.recommendation.course_recommender import CourseRecommender

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories():
    """Create necessary directories"""
    directories = ['data/raw', 'data/processed', 'models', 'results']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def generate_data(config):
    """Generate synthetic data"""
    logger.info("Generating synthetic data...")
    generator = DataGenerator()
    
    customers = generator.generate_customers(config['data']['sample_size'])
    courses = generator.generate_courses(100)
    interactions = generator.generate_interactions(customers, courses, 25000)
    
    # Save data
    customers.to_csv('data/raw/customers.csv', index=False)
    courses.to_csv('data/raw/courses.csv', index=False)
    interactions.to_csv('data/raw/interactions.csv', index=False)
    
    logger.info(f"Data generated and saved: {len(customers)} customers, {len(courses)} courses, {len(interactions)} interactions")
    return customers, courses, interactions

def run_segmentation(customers_df):
    """Run customer segmentation"""
    logger.info("Running customer segmentation...")
    segmenter = CustomerSegmenter(n_segments=4)
    segmented_customers = segmenter.segment_customers(customers_df)
    
    # Save segmented data
    segmented_customers.to_csv('data/processed/segmented_customers.csv', index=False)
    
    # Visualize segments
    segmenter.visualize_segments(segmented_customers, 'results/figures/segment_visualization.png')
    
    logger.info(f"Segmentation complete. Segments: {segmented_customers['segment_name'].unique().tolist()}")
    return segmented_customers, segmenter

def run_lead_scoring(customers_df):
    """Run lead scoring"""
    logger.info("Running lead scoring...")
    scorer = LeadScorer(model_type="xgboost")
    
    # Prepare features
    X, y, feature_cols = scorer.prepare_features(customers_df)
    
    # Train model
    metrics = scorer.train(X, y)
    
    # Save model
    scorer.save_model('models/lead_scorer.joblib')
    
    # Generate predictions for all customers
    predictions = []
    for _, customer in customers_df.iterrows():
        pred = scorer.predict(customer.to_dict())
        predictions.append(pred)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv('results/lead_scores.csv', index=False)
    
    logger.info(f"Lead scoring complete. Model metrics: {metrics}")
    return scorer, pred_df

def run_recommendations(customers_df, courses_df, interactions_df):
    """Run course recommendations"""
    logger.info("Running course recommendations...")
    recommender = CourseRecommender()
    recommender.fit_courses(courses_df, interactions_df)
    
    # Generate recommendations for top customers
    top_customers = customers_df.nlargest(10, 'engagement_score')
    
    recommendations = []
    for _, customer in top_customers.iterrows():
        customer_id = customer['customer_id']
        
        # Get recommendations
        recs = recommender.get_recommendations(
            customer_id=customer_id,
            customer_data={
                'course_views_history': [],  # Would come from interactions
                'segment': customer.get('segment_name', '')
            },
            top_n=5
        )
        
        recommendations.append({
            'customer_id': customer_id,
            'customer_segment': customer.get('segment_name', 'unknown'),
            'recommendations': recs
        })
    
    # Save recommendations
    with open('results/recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info(f"Recommendations generated for {len(recommendations)} customers")
    return recommender, recommendations

def generate_strategic_recommendations(customers_df, predictions_df):
    """Generate strategic recommendations"""
    logger.info("Generating strategic recommendations...")
    
    # Calculate metrics
    conversion_rate = customers_df['converted'].mean() * 100
    
    # Segment performance
    segment_performance = customers_df.groupby('segment_name').agg({
        'converted': ['mean', 'count']
    }).reset_index()
    segment_performance.columns = ['segment', 'conversion_rate', 'count']
    segment_performance['conversion_rate'] = segment_performance['conversion_rate'] * 100
    
    # Social proof analysis
    social_proof_impact = customers_df['converted'].mean() * 100
    
    # Generate recommendations
    recommendations = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_customers': len(customers_df),
            'overall_conversion_rate': conversion_rate,
            'high_value_segment_size': len(customers_df[customers_df['segment_name'] == 'High-Value Customers']),
            'avg_lead_score': predictions_df['score'].mean(),
            'top_segment': segment_performance.iloc[0]['segment'] if not segment_performance.empty else 'N/A'
        },
        'segment_performance': segment_performance.to_dict('records'),
        'strategic_recommendations': [
            {
                'category': 'Marketing Channel Optimization',
                'recommendations': [
                    'Focus 60% of budget on High-Value segment channels',
                    'Increase social media engagement for Engaged Learners',
                    'Optimize email campaigns with personalized content'
                ]
            },
            {
                'category': 'Engagement Improvement',
                'recommendations': [
                    'Implement personalized email sequences for leads',
                    'Create targeted content for each segment',
                    'Develop retention programs for High-Value customers'
                ]
            },
            {
                'category': 'Conversion Rate Enhancement',
                'recommendations': [
                    'A/B test course landing pages',
                    'Add social proof elements (ratings, reviews)',
                    'Implement urgency tactics for top leads'
                ]
            },
            {
                'category': 'Revenue Growth',
                'recommendations': [
                    'Upsell advanced courses to High-Value segment',
                    'Create bundle offers for cross-selling',
                    'Implement referral programs'
                ]
            },
            {
                'category': 'Customer Retention',
                'recommendations': [
                    'Regular check-ins with engaged users',
                    'Personalized course recommendations',
                    'Community building initiatives'
                ]
            }
        ],
        'social_proof_analysis': {
            'impact_rating': social_proof_impact,
            'recommendations': [
                'Showcase top-rated courses prominently',
                'Display social proof on landing pages',
                'Create case studies from successful students'
            ]
        }
    }
    
    # Save recommendations
    with open('results/strategic_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logger.info("Strategic recommendations generated")
    return recommendations

def interactive_mode():
    """Interactive CLI mode"""
    print("\n" + "="*60)
    print("🎯 Lead Scoring & Course Recommendation System")
    print("="*60 + "\n")
    
    print("1. Generate synthetic data")
    print("2. Run full pipeline")
    print("3. Run segmentation only")
    print("4. Run lead scoring only")
    print("5. Run recommendations only")
    print("6. View results")
    print("7. Exit")
    
    choice = input("\nSelect option (1-7): ")
    return choice

def display_results():
    """Display analysis results"""
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    # Load results
    try:
        with open('results/strategic_recommendations.json', 'r') as f:
            recs = json.load(f)
        
        print(f"\n📈 Summary:")
        print(f"  • Total Customers: {recs['summary']['total_customers']}")
        print(f"  • Conversion Rate: {recs['summary']['overall_conversion_rate']:.1f}%")
        print(f"  • Avg Lead Score: {recs['summary']['avg_lead_score']:.1f}")
        
        print("\n🎯 Strategic Recommendations:")
        for rec in recs['strategic_recommendations']:
            print(f"\n  {rec['category']}:")
            for r in rec['recommendations']:
                print(f"    • {r}")
        
        print("\n📊 Segment Performance:")
        for seg in recs['segment_performance']:
            print(f"  • {seg['segment']}: {seg['conversion_rate']:.1f}% conversion ({seg['count']} customers)")
        
    except FileNotFoundError:
        print("No results found. Run the analysis first.")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Lead Scoring & Course Recommendation System')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')
    parser.add_argument('--score', action='store_true', help='Run lead scoring only')
    parser.add_argument('--segment', action='store_true', help='Run segmentation only')
    parser.add_argument('--recommend', action='store_true', help='Run recommendations only')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    config = load_config(args.config)
    create_directories()
    
    if args.interactive:
        choice = interactive_mode()
        if choice == '1':
            generate_data(config)
        elif choice == '2':
            customers, courses, interactions = generate_data(config)
            seg_customers, segmenter = run_segmentation(customers)
            scorer, predictions = run_lead_scoring(seg_customers)
            recommender, recs = run_recommendations(seg_customers, courses, interactions)
            strategies = generate_strategic_recommendations(seg_customers, predictions)
            display_results()
        elif choice == '3':
            customers = pd.read_csv('data/raw/customers.csv')
            run_segmentation(customers)
        elif choice == '4':
            customers = pd.read_csv('data/raw/customers.csv')
            run_lead_scoring(customers)
        elif choice == '5':
            customers = pd.read_csv('data/raw/customers.csv')
            courses = pd.read_csv('data/raw/courses.csv')
            interactions = pd.read_csv('data/raw/interactions.csv')
            run_recommendations(customers, courses, interactions)
        elif choice == '6':
            display_results()
        else:
            print("Exiting...")
    
    elif args.all:
        customers, courses, interactions = generate_data(config)
        seg_customers, segmenter = run_segmentation(customers)
        scorer, predictions = run_lead_scoring(seg_customers)
        recommender, recs = run_recommendations(seg_customers, courses, interactions)
        strategies = generate_strategic_recommendations(seg_customers, predictions)
        display_results()
    
    elif args.score:
        customers = pd.read_csv('data/raw/customers.csv')
        run_lead_scoring(customers)
    
    elif args.segment:
        customers = pd.read_csv('data/raw/customers.csv')
        run_segmentation(customers)
    
    elif args.recommend:
        customers = pd.read_csv('data/raw/customers.csv')
        courses = pd.read_csv('data/raw/courses.csv')
        interactions = pd.read_csv('data/raw/interactions.csv')
        run_recommendations(customers, courses, interactions)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
