import os
import re
import urllib.request
import json

# Fallback Physics Textbook Database (Real NCERT Physics Part 1 text and formulas for offline sandbox testing)
MOCK_PHYSICS_DATABASE = [
    {
        "chapter": "Chapter 1: Electric Charges and Fields",
        "section": "1.3 Coulomb's Law",
        "page": 9,
        "text": "Coulomb's Law states that the mutual electrostatic force between two point charges q1 and q2 is proportional to the product of the charges and inversely proportional to the square of the distance r between them. The force acts along the line joining the two charges. Mathematically, the magnitude of the force F is given by F = (1 / (4 * pi * epsilon_0)) * (q1 * q2 / r^2), where epsilon_0 is the permittivity of free space. The value of 1 / (4 * pi * epsilon_0) is approximately 9 * 10^9 N m^2 C^-2.",
        "formulas": ["F = (1 / (4 * pi * \\varepsilon_0)) * (q_1 * q_2 / r^2)", "\\varepsilon_0 = 8.854 \\times 10^{-12} C^2 N^{-1} m^{-2}"]
    },
    {
        "chapter": "Chapter 1: Electric Charges and Fields",
        "section": "1.9 Electric Flux",
        "page": 25,
        "text": "Electric flux is a measure of the number of electric field lines passing through a given surface. For a uniform electric field E passing through a flat area A, the electric flux Phi is defined as Phi = E * A * cos(theta), where theta is the angle between the electric field lines and the normal (perpendicular) to the surface. The SI unit of electric flux is Newton meters squared per Coulomb (N m^2 C^-1). It is a scalar quantity.",
        "formulas": ["\\Phi_E = \\mathbf{E} \\cdot \\mathbf{A} = E A \\cos(\\theta)", "\\Phi_E = \\int \\mathbf{E} \\cdot d\\mathbf{A}"]
    },
    {
        "chapter": "Chapter 1: Electric Charges and Fields",
        "section": "1.10 Gauss's Law",
        "page": 33,
        "text": "Gauss's Law states that the total electric flux through any closed surface is equal to 1 / epsilon_0 times the net charge enclosed by the surface. Mathematically, the closed surface integral of the electric field E over an area element dA is equal to q / epsilon_0, where q is the total charge enclosed. This law is extremely useful for calculating the electric field of symmetric charge distributions, such as spheres, cylinders, and infinite sheets.",
        "formulas": ["\\oint \\mathbf{E} \\cdot d\\mathbf{A} = \\frac{q_{enclosed}}{\\varepsilon_0}"]
    },
    {
        "chapter": "Chapter 2: Electrostatic Potential and Capacitance",
        "section": "2.2 Electrostatic Potential",
        "page": 53,
        "text": "Electrostatic potential at any point in an electric field is defined as the work done by an external agent in bringing a unit positive charge from infinity to that point without acceleration. The potential V due to a point charge q at a distance r is given by V = q / (4 * pi * epsilon_0 * r). Potential is a scalar quantity and its SI unit is the Volt (V) or Joules per Coulomb (J C^-1).",
        "formulas": ["V = \\frac{q}{4 \\pi \\varepsilon_0 r}", "W = q V"]
    },
    {
        "chapter": "Chapter 2: Electrostatic Potential and Capacitance",
        "section": "2.8 Capacitors and Capacitance",
        "page": 74,
        "text": "A capacitor is a system of two conductors separated by an insulator. Capacitance C is the ratio of the magnitude of charge Q on either conductor to the potential difference V between them, so C = Q / V. The SI unit of capacitance is the Farad (F). For a parallel plate capacitor in vacuum with plate area A and separation distance d, the capacitance is given by C = epsilon_0 * A / d. Inserting a dielectric between the plates increases the capacitance by a factor of K, the dielectric constant.",
        "formulas": ["C = \\frac{Q}{V}", "C = \\frac{\\varepsilon_0 A}{d}", "C = \\frac{K \\varepsilon_0 A}{d}"]
    },
    {
        "chapter": "Chapter 3: Current Electricity",
        "section": "3.2 Electric Current",
        "page": 93,
        "text": "Electric current is defined as the rate at which charge flows through a cross-section of a conductor. If a net charge Q passes through any cross-section of a conductor in time t, the current I is I = Q / t. The SI unit of current is the Ampere (A). In conductors, current is carried by the drift of free electrons under the influence of an electric field. The drift velocity v_d is proportional to the applied field.",
        "formulas": ["I = \\frac{d Q}{d t}", "I = n A e v_d"]
    },
    {
        "chapter": "Chapter 3: Current Electricity",
        "section": "3.3 Ohm's Law",
        "page": 95,
        "text": "Ohm's Law states that the current I flowing through a conductor is directly proportional to the potential difference V across its ends, provided physical conditions like temperature remain constant. Mathematically, V = I * R, where R is the resistance of the conductor. The SI unit of resistance is the Ohm. Resistance R depends on length l and cross-sectional area A of the conductor: R = rho * l / A, where rho is the resistivity of the material.",
        "formulas": ["V = I R", "R = \\rho \\frac{l}{A}", "\\mathbf{J} = \\sigma \\mathbf{E}"]
    },
    {
        "chapter": "Chapter 4: Moving Charges and Magnetism",
        "section": "4.5 Ampere's Circuital Law",
        "page": 147,
        "text": "Ampere's Circuital Law states that the line integral of the magnetic field B around any closed loop is equal to mu_0 times the total current passing through the surface enclosed by the loop. Mathematically, the closed line integral of B dot dl is equal to mu_0 * I, where I is the enclosed current. This law is analogous to Gauss's Law in electrostatics and is used to find magnetic fields of high symmetry.",
        "formulas": ["\\oint \\mathbf{B} \\cdot d\\mathbf{l} = \\mu_0 I"]
    },
    {
        "chapter": "Chapter 4: Moving Charges and Magnetism",
        "section": "4.8 Biot-Savart Law",
        "page": 143,
        "text": "The Biot-Savart Law gives the magnetic field dB produced by a small current element dl carrying current I at a distance r. The magnitude of dB is proportional to the current I, length dl, sine of the angle theta between the element and position vector r, and inversely proportional to the square of r. Mathematically, dB = (mu_0 / 4 * pi) * (I * dl * sin(theta) / r^2). The constant mu_0 is the permeability of free space.",
        "formulas": ["d\\mathbf{B} = \\frac{\\mu_0}{4 \\pi} \\frac{I d\\mathbf{l} \\times \\hat{\\mathbf{r}}}{r^2}"]
    },
    {
        "chapter": "Chapter 6: Electromagnetic Induction",
        "section": "6.3 Faraday's Law of Induction",
        "page": 207,
        "text": "Faraday's Law of Electromagnetic Induction states that the magnitude of the induced electromotive force (emf) in a circuit is equal to the time rate of change of magnetic flux through the circuit. The magnetic flux Phi_B through a surface is given by the integral of B dot dA. The induced emf epsilon is given by epsilon = - d(Phi_B) / dt. The negative sign represents the direction of the induced emf (Lenz's Law).",
        "formulas": ["\\varepsilon = -\\frac{d \\Phi_B}{d t}", "\\Phi_B = \\int \\mathbf{B} \\cdot d\\mathbf{A}"]
    },
    {
        "chapter": "Chapter 6: Electromagnetic Induction",
        "section": "6.4 Lenz's Law",
        "page": 210,
        "text": "Lenz's Law states that the direction of the induced current is such that it opposes the change in magnetic flux that produced it. Lenz's Law is a consequence of the law of conservation of energy. When a magnet is pushed towards a coil, the induced current creates a magnetic field that repels the incoming magnet, requiring work to be done. This work is converted into electrical energy in the coil.",
        "formulas": ["\\varepsilon = -\\frac{d \\Phi_B}{d t}"]
    }
]

NCERT_PDF_URL = "https://ncert.nic.in/textbook/pdf/leph101.pdf" # Chapter 1 PDF for testing

def download_physics_pdf(output_path):
    """
    Attempts to download a sample chapter PDF from the NCERT website.
    If offline, catches the error and logs a message.
    """
    if os.path.exists(output_path):
        return True
        
    print(f"Attempting to download NCERT Physics PDF to: {output_path}")
    try:
        # Configure headers to look like a browser request to avoid NCERT scraping blocks
        req = urllib.request.Request(
            NCERT_PDF_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Download failed (Offline sandbox or NCERT server down: {e}). Using local fallback database.")
        return False

def parse_pdf(pdf_path):
    """
    Parses the target PDF page by page.
    If pypdf/pdfplumber is available, extracts text and attempts to isolate formulas.
    Otherwise, returns the structured mock database of real physics chapters.
    """
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}. Using fallback database.")
        return MOCK_PHYSICS_DATABASE
        
    chunks = []
    use_pdfplumber = False
    
    try:
        import pdfplumber
        use_pdfplumber = True
    except ImportError:
        pass
        
    if use_pdfplumber:
        print(f"Parsing PDF {pdf_path} using pdfplumber...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Basic parsing loop
                current_chapter = "Chapter 1: Electric Charges and Fields"
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    page_num = i + 1
                    
                    # Look for chapter headers in text
                    chap_match = re.search(r'(Chapter\s+\d+|CHAPTER\s+\d+)', text)
                    if chap_match:
                        current_chapter = chap_match.group(1).strip()
                        
                    # Locate sections
                    section = "General Section"
                    sec_match = re.search(r'(\d+\.\d+\s+[A-Za-z\s]+)', text)
                    if sec_match:
                        section = sec_match.group(1).strip()
                        
                    # Isolate formulas (e.g. look for equations with = signs or mathematical variables)
                    formulas = []
                    lines = text.split('\n')
                    for line in lines:
                        if '=' in line and len(line) < 100:
                            # Heuristic for formula lines
                            if any(var in line for var in ['F', 'E', 'Phi', 'V', 'C', 'Q', 'I', 'R', 'B', 'epsilon_0', 'mu_0']):
                                formulas.append(line.strip())
                                
                    if len(text.strip()) > 100:
                        chunks.append({
                            "chapter": current_chapter,
                            "section": section,
                            "page": page_num,
                            "text": text.strip(),
                            "formulas": formulas
                        })
            print(f"Successfully extracted {len(chunks)} pages from PDF.")
            return chunks
        except Exception as e:
            print(f"Failed to parse PDF: {e}. Falling back to default database.")
            return MOCK_PHYSICS_DATABASE
    else:
        print("pdfplumber not installed. Using local fallback database.")
        return MOCK_PHYSICS_DATABASE

def get_chunks():
    """
    Main ingestion entrypoint.
    Downloads sample if missing and returns structured text chunks.
    """
    pdf_dir = 'hybrid_rag/data'
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, 'ncert_physics_part1.pdf')
    
    # Download sample PDF if missing (will run online)
    download_physics_pdf(pdf_path)
    
    # Parse PDF (will load mock database if offline/missing)
    chunks = parse_pdf(pdf_path)
    return chunks

if __name__ == '__main__':
    chunks = get_chunks()
    print(f"Total chunks extracted: {len(chunks)}")
    print(f"First chunk sample:\n{json.dumps(chunks[0], indent=2)}")
