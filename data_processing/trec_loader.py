import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DATA_DIR = 'data/erisk25-t1-dataset'
OUTPUT_DIR = "data/erisk25-t1-processed"

def read_trec(file_path):
    file_id = os.path.basename(file_path).split('.')[0]

    # Wrap and clean content, then parse with BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as f:
        wrapped_content = f"<ROOT>{f.read()}</ROOT>"

    # Parse with BeautifulSoup
    soup = BeautifulSoup(wrapped_content, 'xml')
    docs = soup.find_all('DOC')

    # Extract document data
    doc_data = []
    for doc in docs:
        doc_data.append({
            'doc_no': doc.DOCNO.text if doc.DOCNO else '',
            'doc_pre': doc.PRE.text if doc.PRE else '',
            'doc_text': doc.TEXT.text if doc.TEXT else '',
            'doc_post': doc.POST.text if doc.POST else ''
        })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(doc_data)
    output_path = f'{OUTPUT_DIR}/{file_id}.csv'
    df.to_csv(output_path, index=False)
    
    
def process_file(file):
    try:
        return read_trec(os.path.join(DATA_DIR, file))
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None
    
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = os.listdir(DATA_DIR)

    print(os.cpu_count())
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in tqdm(as_completed(futures), total=len(files)):
            future.result()
    
            
if __name__ == '__main__':
    print("Converting TREC files to CSV...")
    main()