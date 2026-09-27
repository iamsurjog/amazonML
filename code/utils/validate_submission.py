import os
import csv
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matching', required=True, help="Path to matching_results.tsv")
    parser.add_argument('--candidate', required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument('--test-dir', required=True, help="Path to test dataset directory")
    return parser.parse_args()

def load_test_entities(test_dir):
    s1_entities = set()
    s2_s3_entities = set()
    
    for filename in os.listdir(test_dir):
        if not filename.endswith('.tsv'): continue
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row['entity_id']
                if eid.startswith('S1-'):
                    s1_entities.add(eid)
                elif eid.startswith('S2-') or eid.startswith('S3-'):
                    s2_s3_entities.add(eid)
    return s1_entities, s2_s3_entities

def validate_file(filepath, required_columns, s1_test_set, s2_s3_test_set, is_candidate_file=False):
    errors = []
    seen_s1 = set()
    
    if not os.path.exists(filepath):
        return [f"File not found: {filepath}"], {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader, None)
        
        if headers != required_columns:
            errors.append(f"Invalid headers in {filepath}. Expected {required_columns}, got {headers}")
            return errors, {}

        file_data = {}
        for line_num, row in enumerate(reader, start=2):
            if len(row) != 2:
                errors.append(f"Line {line_num}: Must have exactly two columns.")
                continue
                
            s1_id, matched_str = row
            
            # Rule: Every Source 1 entity must have exactly one row / No duplicate entity IDs
            if s1_id in seen_s1:
                errors.append(f"Line {line_num}: Duplicate source1_entity_id found: {s1_id}")
            seen_s1.add(s1_id)
            
            # Rule: Leave empty for singletons
            matched_ids = [m.strip() for m in matched_str.split(',')] if matched_str.strip() else []
            
            # Rule: No duplicate entity IDs within a single ID list
            if len(matched_ids) != len(set(matched_ids)):
                errors.append(f"Line {line_num}: Duplicate IDs found in the matched list for {s1_id}")
            
            for m_id in matched_ids:
                # Rule: Must only reference entities from Source 2 or Source 3
                if not (m_id.startswith('S2-') or m_id.startswith('S3-')):
                    errors.append(f"Line {line_num}: Invalid target ID {m_id}. Self-matches to Source 1 are rejected.")
                
                # Rule: IDs must exist in the test set
                if m_id not in s2_s3_test_set:
                    errors.append(f"Line {line_num}: Target ID {m_id} does not exist in the test set.")
            
            file_data[s1_id] = set(matched_ids)

    # Rule: Every Source 1 entity in the test set must appear
    missing_s1 = s1_test_set - seen_s1
    if missing_s1:
        errors.append(f"Missing {len(missing_s1)} Source 1 entities from the test set in {filepath}.")

    return errors, file_data

def main():
    args = parse_args()
    errors = []
    
    print("Loading test set entities...")
    s1_test, s2_s3_test = load_test_entities(args.test_dir)
    
    print(f"Validating {args.candidate}...")
    cand_errors, cand_data = validate_file(
        args.candidate, 
        ['source1_entity_id', 'candidate_entity_ids'], 
        s1_test, s2_s3_test, 
        is_candidate_file=True
    )
    errors.extend(cand_errors)
    
    print(f"Validating {args.matching}...")
    match_errors, match_data = validate_file(
        args.matching, 
        ['source1_entity_id', 'matched_entity_ids'], 
        s1_test, s2_s3_test
    )
    errors.extend(match_errors)
    
    # Rule: Your final matches should be a subset of your candidates
    print("Checking if matches are a subset of candidates...")
    for s1_id, matches in match_data.items():
        if s1_id in cand_data:
            invalid_matches = matches - cand_data[s1_id]
            if invalid_matches:
                errors.append(f"Matches for {s1_id} contain IDs not present in candidates: {invalid_matches}")

    if errors:
        print("\nVALIDATION FAILED. Fix the following issues (Exit 1):")
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        sys.exit(1)
    else:
        print("\nPASS (Exit 0): Files are safe to submit.")
        sys.exit(0)

if __name__ == '__main__':
    main()
