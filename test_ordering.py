#!/usr/bin/env python3
"""
Test script to verify standardized group ordering works correctly
"""

def standardize_group_order(groups):
    """
    Standardize the order of groups to ensure consistent chart rendering and caching.
    Sort alphabetically (a-z, then 0-9), with "not provided" always at the end.
    """
    def sort_key(item):
        item_str = str(item).lower()
        if item_str == "not provided":
            return "zzz_not_provided"  # Ensures it comes last
        return item_str
    
    return sorted(groups, key=sort_key)

# Test cases
print("=== Testing Standardized Group Ordering ===")

test_cases = [
    # Test case 1: Basic alphabet and numbers
    (['C', 'A', 'B', 'not provided', '2', '1'], ['1', '2', 'A', 'B', 'C', 'not provided']),
    
    # Test case 2: Real microbiome categories
    (['Mental illness', 'not provided', 'ASD', 'Healthy'], ['ASD', 'Healthy', 'Mental illness', 'not provided']),
    
    # Test case 3: Age categories
    (['51-70 years', '18-30 years', 'not provided', '31-50 years'], ['18-30 years', '31-50 years', '51-70 years', 'not provided']),
    
    # Test case 4: Sex categories
    (['Male', 'Female', 'not provided'], ['Female', 'Male', 'not provided']),
    
    # Test case 5: Mixed case and special characters
    (['sample_type_B', 'Sample_Type_A', 'not provided', '1st_group'], ['1st_group', 'sample_type_a', 'sample_type_b', 'not provided']),
]

for i, (input_groups, expected) in enumerate(test_cases, 1):
    result = standardize_group_order(input_groups)
    print(f"\nTest Case {i}:")
    print(f"  Input:    {input_groups}")
    print(f"  Output:   {result}")
    print(f"  Expected: {expected}")
    print(f"  Status:   {'✅ PASS' if result == expected else '❌ FAIL'}")

print("\n=== Cache Key Benefits ===")
# Demonstrate cache key consistency
groups1 = ['C', 'A', 'B']
groups2 = ['B', 'C', 'A'] 
groups3 = ['A', 'B', 'C']

standardized1 = standardize_group_order(groups1)
standardized2 = standardize_group_order(groups2)
standardized3 = standardize_group_order(groups3)

print(f"Different input orders:")
print(f"  {groups1} -> {standardized1}")
print(f"  {groups2} -> {standardized2}")
print(f"  {groups3} -> {standardized3}")
print(f"All produce same result: {'✅ YES' if standardized1 == standardized2 == standardized3 else '❌ NO'}")
