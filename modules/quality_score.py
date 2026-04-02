def compute_quality_score(missing, duplicates, outliers, inconsistencies):

    # Apply aggressive multipliers to penalties. 
    # Example: 16% duplicates in a small dataset drastically impacts model variance.
    completeness = max(0, 100 - (missing["missing_percent_mean"] * 4))
    uniqueness = max(0, 100 - (duplicates["duplicate_percent"] * 3))
    outlier_score = max(0, 100 - (outliers["outlier_ratio"] * 2))

    consistency = max(0, 100 - inconsistencies * 10)

    score = (
        completeness * 0.3 +
        uniqueness * 0.3 +       # Increased weight for uniqueness
        outlier_score * 0.2 +    # Reduced slightly
        consistency * 0.2
    ) / 10

    return round(score, 2)