import requests
import json
from pathlib import Path
from rich import print

# SEC requires a User-Agent header with your contact info
HEADERS = {
    'User-Agent': 'Andrew Hall andrewmartinhall2@gmail.com'  # Replace with your info
}

BASE_URL = 'https://data.sec.gov'
def get_company_filings(cik: str, filing_types=['10-K', '10-Q']):
    """
    Get filings for a company using their CIK number.
    CIK should be 10 digits with leading zeros (e.g., '0000320193' for Apple)
    """
    # Ensure CIK is 10 digits with leading zeros
    cik = cik.zfill(10)
    
    url = f'{BASE_URL}/submissions/CIK{cik}.json'
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    data = response.json()
    print(data)
    filings = data['filings']['recent']
    
    # Filter for specific filing types
    results = []
    for i in range(len(filings['form'])):
        if filings['form'][i] in filing_types:
            results.append({
                'form': filings['form'][i],
                'filing_date': filings['filingDate'][i],
                'accession_number': filings['accessionNumber'][i],
                'primary_document': filings['primaryDocument'][i]
            })
    
    return results

def download_filing(cik, accession_number, primary_document, save_path):
    """
    Download the actual filing document (HTML or text format).
    """
    # Remove dashes from accession number for URL
    accession_no_dashes = accession_number.replace('-', '')
    cik = str(int(cik))  # Remove leading zeros for the URL path
    
    # Construct the document URL - note: uses www.sec.gov, not data.sec.gov
    url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_document}'
    
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    return save_path

# Example usage
apple_cik = '0000320193'
filings = get_company_filings(apple_cik)

print(filings)

# Example usage
filing = filings[0]
download_filing(
    apple_cik,
    filing['accession_number'],
    filing['primary_document'],
    f"apple_{filing['form']}_{filing['filing_date']}.html"
)

