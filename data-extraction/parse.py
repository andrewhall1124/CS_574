from bs4 import BeautifulSoup
import re

def extract_risk_factors(html_file_path, output_file_path):
    """
    Extract Risk Factors section from SEC 10-K HTML filing.
    Extracts text between Item 1A (Risk Factors) and Item 1B (Unresolved Staff Comments).
    """
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all text elements
    all_elements = soup.find_all(['div', 'p', 'span'])
    
    # Find the starting point (Item 1A - Risk Factors)
    start_index = None
    end_index = None
    
    for i, element in enumerate(all_elements):
        text = element.get_text(strip=True)
        
        # Look for Item 1A Risk Factors - must be standalone (not part of table of contents)
        if start_index is None:
            if re.match(r'^Item\s*1A\.?\s*Risk\s*Factors\s*$', text, re.IGNORECASE):
                start_index = i
                print(f"Found start at index {i}: '{text}'")
        
        # Look for Item 1B (end of risk factors) - must be standalone
        elif start_index is not None and i > start_index + 5:  # Must be at least 5 elements after start
            if re.match(r'^Item\s*1B\.?\s*Unresolved\s*Staff\s*Comments\s*$', text, re.IGNORECASE):
                end_index = i
                print(f"Found end at index {i}: '{text}'")
                break
    
    if start_index is None:
        print("Could not find Item 1A (Risk Factors)")
        return None
    
    if end_index is None:
        print("Warning: Could not find Item 1B, extracting to end of document")
        end_index = len(all_elements)
    
    # Extract text from elements between start and end
    risk_text_parts = []
    
    for i in range(start_index + 1, end_index):  # Start from element after Item 1A header
        element = all_elements[i]
        text = element.get_text(strip=True)
        
        # Skip empty elements
        if not text or len(text) < 10:
            continue
        
        # Skip page numbers/footers (e.g., "Apple Inc. | 2025 Form 10-K | 5")
        if re.search(r'Apple Inc\.\s*\|\s*\d{4}\s*Form\s*10-K\s*\|\s*\d+', text):
            continue
        
        # Skip headers that are just section labels
        if re.match(r'^(Macroeconomic and Industry Risks|Business Risks|Legal and Regulatory Compliance Risks|Financial Risks|General Risks)$', text):
            risk_text_parts.append(f"\n=== {text} ===\n")
            continue
        
        # Skip if this element contains other divs (to avoid duplicate text)
        if element.name == 'div' and element.find('div'):
            continue
        
        # Skip if element is a span and parent already added
        if element.name == 'span':
            parent = element.parent
            if parent and parent.name in ['div', 'p']:
                # Check if parent text is substantially the same
                parent_text = parent.get_text(strip=True)
                if text in parent_text and len(parent_text) - len(text) < 50:
                    continue
        
        # Add the text
        risk_text_parts.append(text)
    
    # Join all text parts
    full_text = '\n\n'.join(risk_text_parts)
    
    # Save to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\nSuccessfully extracted Risk Factors section")
    print(f"Total characters: {len(full_text)}")
    print(f"Total paragraphs: {len(risk_text_parts)}")
    print(f"Saved to: {output_file_path}")
    
    return full_text


if __name__ == "__main__":
    # File paths
    html_file = 'apple_10-K_2025-10-31.html'
    output_file = 'apple_risk_factors.txt'
    
    # Extract risk factors
    risk_text = extract_risk_factors(html_file, output_file)
    
    # Print preview
    if risk_text:
        print("\n" + "="*80)
        print("PREVIEW OF EXTRACTED TEXT (first 1000 characters):")
        print("="*80)
        print(risk_text[:1000])
        print("...")