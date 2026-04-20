"""
Synthetic Dataset Generator for Learning Disabilities Detection
================================================================
Generates a realistic synthetic dataset based on published research
on learning disability indicators in children.

Disability Types:
    0 - No Disability
    1 - Dyslexia (reading difficulty)
    2 - Dysgraphia (writing difficulty)
    3 - Dyscalculia (math difficulty)
    4 - ADHD (attention deficit/hyperactivity)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_SAMPLES = 5000

# Distribution: 40% No Disability, 15% each for the 4 types
CLASS_DISTRIBUTION = {
    0: int(NUM_SAMPLES * 0.40),  # No Disability - 2000
    1: int(NUM_SAMPLES * 0.15),  # Dyslexia - 750
    2: int(NUM_SAMPLES * 0.15),  # Dysgraphia - 750
    3: int(NUM_SAMPLES * 0.15),  # Dyscalculia - 750
    4: int(NUM_SAMPLES * 0.15),  # ADHD - 750
}

PARENT_EDUCATION_LEVELS = ['High School', 'Bachelors', 'Masters', 'PhD']


def generate_class_data(label, n):
    """Generate feature data for a specific disability class."""
    data = {}

    # Age (5-15 years) - uniform across all classes
    data['age'] = np.random.randint(5, 16, n)

    # Gender (0=Female, 1=Male)
    if label == 4:  # ADHD more common in males
        data['gender'] = np.random.choice([0, 1], n, p=[0.3, 0.7])
    else:
        data['gender'] = np.random.choice([0, 1], n, p=[0.5, 0.5])

    # === Academic Scores ===
    if label == 0:  # No Disability
        data['reading_score'] = np.clip(np.random.normal(75, 12, n), 0, 100)
        data['writing_score'] = np.clip(np.random.normal(73, 12, n), 0, 100)
        data['math_score'] = np.clip(np.random.normal(74, 12, n), 0, 100)
    elif label == 1:  # Dyslexia - LOW reading
        data['reading_score'] = np.clip(np.random.normal(35, 12, n), 0, 100)
        data['writing_score'] = np.clip(np.random.normal(50, 14, n), 0, 100)
        data['math_score'] = np.clip(np.random.normal(65, 14, n), 0, 100)
    elif label == 2:  # Dysgraphia - LOW writing
        data['reading_score'] = np.clip(np.random.normal(60, 14, n), 0, 100)
        data['writing_score'] = np.clip(np.random.normal(30, 12, n), 0, 100)
        data['math_score'] = np.clip(np.random.normal(62, 14, n), 0, 100)
    elif label == 3:  # Dyscalculia - LOW math
        data['reading_score'] = np.clip(np.random.normal(65, 14, n), 0, 100)
        data['writing_score'] = np.clip(np.random.normal(62, 14, n), 0, 100)
        data['math_score'] = np.clip(np.random.normal(30, 12, n), 0, 100)
    elif label == 4:  # ADHD - moderate across board, low attention
        data['reading_score'] = np.clip(np.random.normal(55, 15, n), 0, 100)
        data['writing_score'] = np.clip(np.random.normal(52, 15, n), 0, 100)
        data['math_score'] = np.clip(np.random.normal(54, 15, n), 0, 100)

    # === Cognitive Scores ===
    if label == 0:
        data['attention_span'] = np.clip(np.random.normal(7.5, 1.2, n), 1, 10).astype(int)
        data['memory_score'] = np.clip(np.random.normal(75, 10, n), 0, 100)
        data['processing_speed'] = np.clip(np.random.normal(72, 10, n), 0, 100)
    elif label == 1:  # Dyslexia
        data['attention_span'] = np.clip(np.random.normal(6, 1.5, n), 1, 10).astype(int)
        data['memory_score'] = np.clip(np.random.normal(55, 12, n), 0, 100)
        data['processing_speed'] = np.clip(np.random.normal(50, 12, n), 0, 100)
    elif label == 2:  # Dysgraphia
        data['attention_span'] = np.clip(np.random.normal(6.5, 1.5, n), 1, 10).astype(int)
        data['memory_score'] = np.clip(np.random.normal(60, 12, n), 0, 100)
        data['processing_speed'] = np.clip(np.random.normal(55, 12, n), 0, 100)
    elif label == 3:  # Dyscalculia
        data['attention_span'] = np.clip(np.random.normal(6, 1.5, n), 1, 10).astype(int)
        data['memory_score'] = np.clip(np.random.normal(52, 12, n), 0, 100)
        data['processing_speed'] = np.clip(np.random.normal(48, 12, n), 0, 100)
    elif label == 4:  # ADHD - very low attention
        data['attention_span'] = np.clip(np.random.normal(3, 1.2, n), 1, 10).astype(int)
        data['memory_score'] = np.clip(np.random.normal(58, 14, n), 0, 100)
        data['processing_speed'] = np.clip(np.random.normal(60, 14, n), 0, 100)

    # === Behavioral Scores ===
    if label == 0:
        data['hyperactivity_score'] = np.clip(np.random.normal(3, 1.2, n), 1, 10).astype(int)
        data['impulsivity_score'] = np.clip(np.random.normal(3, 1.2, n), 1, 10).astype(int)
        data['social_skills'] = np.clip(np.random.normal(7.5, 1.2, n), 1, 10).astype(int)
    elif label == 1:
        data['hyperactivity_score'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['impulsivity_score'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['social_skills'] = np.clip(np.random.normal(6, 1.5, n), 1, 10).astype(int)
    elif label == 2:
        data['hyperactivity_score'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['impulsivity_score'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['social_skills'] = np.clip(np.random.normal(6, 1.5, n), 1, 10).astype(int)
    elif label == 3:
        data['hyperactivity_score'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['impulsivity_score'] = np.clip(np.random.normal(4.5, 1.5, n), 1, 10).astype(int)
        data['social_skills'] = np.clip(np.random.normal(5.5, 1.5, n), 1, 10).astype(int)
    elif label == 4:  # ADHD - HIGH hyperactivity & impulsivity
        data['hyperactivity_score'] = np.clip(np.random.normal(8, 1.2, n), 1, 10).astype(int)
        data['impulsivity_score'] = np.clip(np.random.normal(8, 1.2, n), 1, 10).astype(int)
        data['social_skills'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)

    # === Motor & Language Skills ===
    if label == 0:
        data['fine_motor_skills'] = np.clip(np.random.normal(75, 10, n), 0, 100)
        data['phonological_awareness'] = np.clip(np.random.normal(78, 10, n), 0, 100)
        data['vocabulary_score'] = np.clip(np.random.normal(76, 10, n), 0, 100)
    elif label == 1:  # Dyslexia - LOW phonological awareness
        data['fine_motor_skills'] = np.clip(np.random.normal(65, 12, n), 0, 100)
        data['phonological_awareness'] = np.clip(np.random.normal(30, 12, n), 0, 100)
        data['vocabulary_score'] = np.clip(np.random.normal(45, 14, n), 0, 100)
    elif label == 2:  # Dysgraphia - LOW fine motor
        data['fine_motor_skills'] = np.clip(np.random.normal(28, 12, n), 0, 100)
        data['phonological_awareness'] = np.clip(np.random.normal(60, 14, n), 0, 100)
        data['vocabulary_score'] = np.clip(np.random.normal(60, 14, n), 0, 100)
    elif label == 3:  # Dyscalculia
        data['fine_motor_skills'] = np.clip(np.random.normal(60, 14, n), 0, 100)
        data['phonological_awareness'] = np.clip(np.random.normal(62, 14, n), 0, 100)
        data['vocabulary_score'] = np.clip(np.random.normal(60, 14, n), 0, 100)
    elif label == 4:  # ADHD
        data['fine_motor_skills'] = np.clip(np.random.normal(55, 15, n), 0, 100)
        data['phonological_awareness'] = np.clip(np.random.normal(58, 15, n), 0, 100)
        data['vocabulary_score'] = np.clip(np.random.normal(55, 15, n), 0, 100)

    # === Environmental Factors ===
    if label == 0:
        data['parent_education'] = np.random.choice(
            PARENT_EDUCATION_LEVELS, n, p=[0.2, 0.35, 0.30, 0.15])
        data['family_history_ld'] = np.random.choice([0, 1], n, p=[0.85, 0.15])
        data['sleep_hours'] = np.clip(np.random.normal(8.5, 1.0, n), 4, 12)
        data['screen_time'] = np.clip(np.random.normal(2.5, 1.0, n), 0, 8)
    else:
        # Higher probability of family history for LD cases
        data['parent_education'] = np.random.choice(
            PARENT_EDUCATION_LEVELS, n, p=[0.30, 0.35, 0.25, 0.10])
        data['family_history_ld'] = np.random.choice([0, 1], n, p=[0.55, 0.45])
        data['sleep_hours'] = np.clip(np.random.normal(7.5, 1.2, n), 4, 12)
        data['screen_time'] = np.clip(np.random.normal(3.5, 1.5, n), 0, 8)

    # === Classroom Behavior ===
    if label == 0:
        data['class_participation'] = np.clip(np.random.normal(7.5, 1.2, n), 1, 10).astype(int)
        data['emotional_stability'] = np.clip(np.random.normal(7.5, 1.2, n), 1, 10).astype(int)
    elif label == 4:  # ADHD - low participation & stability
        data['class_participation'] = np.clip(np.random.normal(4, 1.5, n), 1, 10).astype(int)
        data['emotional_stability'] = np.clip(np.random.normal(3.5, 1.5, n), 1, 10).astype(int)
    else:  # Other LDs
        data['class_participation'] = np.clip(np.random.normal(5, 1.5, n), 1, 10).astype(int)
        data['emotional_stability'] = np.clip(np.random.normal(5.5, 1.5, n), 1, 10).astype(int)

    # Target
    data['disability_type'] = label

    return pd.DataFrame(data)


def main():
    """Generate and save the full dataset."""
    print("=" * 60)
    print("  Learning Disabilities Dataset Generator")
    print("=" * 60)

    frames = []
    labels = {
        0: "No Disability",
        1: "Dyslexia",
        2: "Dysgraphia",
        3: "Dyscalculia",
        4: "ADHD"
    }

    for label, count in CLASS_DISTRIBUTION.items():
        print(f"  Generating {count} samples for: {labels[label]}...")
        df = generate_class_data(label, count)
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    # Shuffle the dataset
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Round float columns to 2 decimal places
    float_cols = dataset.select_dtypes(include=['float64']).columns
    dataset[float_cols] = dataset[float_cols].round(2)

    # Save
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', 'learning_disabilities.csv')
    dataset.to_csv(filepath, index=False)

    print(f"\n  Dataset saved to: {filepath}")
    print(f"  Total samples: {len(dataset)}")
    print(f"\n  Class Distribution:")
    for label, name in labels.items():
        count = (dataset['disability_type'] == label).sum()
        print(f"    {name}: {count} ({count/len(dataset)*100:.1f}%)")

    print(f"\n  Features: {list(dataset.columns[:-1])}")
    print(f"  Shape: {dataset.shape}")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    main()
