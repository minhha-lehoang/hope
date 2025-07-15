import os
import pandas as pd
from glob import glob

RELEVANT_THRESHOLD = 0.1


def filter_relevant_predictions(
    input_dir='outputs/extractor_predictions',
    output_file=f'outputs/relevant_predictions_{RELEVANT_THRESHOLD}.csv'
):
    """
    Reads extractor output CSVs, filters rows where any query score exceeds threshold,
    and saves all matching rows into one CSV file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    csv_files = glob(os.path.join(input_dir, '*.csv'))
    if not csv_files:
        print(f"No prediction files found in {input_dir}")
        return

    header_written = False
    total_saved = 0
    for csv_file in csv_files:
        for chunk in pd.read_csv(csv_file, chunksize=10000):
            # Identify query columns
            query_cols = [c for c in chunk.columns if c.startswith('query_')]
            # Filter rows with any score > threshold
            mask = chunk[query_cols].gt(RELEVANT_THRESHOLD).any(axis=1)
            filtered = chunk[mask]
            if not filtered.empty:
                # Write header once
                if not header_written:
                    filtered.to_csv(output_file, index=False, mode='w')
                    header_written = True
                else:
                    filtered.to_csv(output_file, index=False, header=False, mode='a')
                total_saved += len(filtered)

    print(f"Saved {total_saved} relevant rows to {output_file}")


if __name__ == '__main__':
    filter_relevant_predictions()