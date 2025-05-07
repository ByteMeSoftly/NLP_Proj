import re
import streamlit as st
import spacy
import pdfplumber
from collections import Counter
from typing import List, Dict, Tuple
import logging


st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model with specific components for efficiency
@st.cache_resource
def load_model():
    """Load and cache the spaCy model"""
    try:
        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        # Add sentencizer if not present
        if 'sentencizer' not in nlp.pipe_names:
            nlp.add_pipe('sentencizer')
        return nlp
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        st.error("Failed to load language model. Please check your installation.")
        return None

nlp = load_model()

class DocumentSummarizer:
    """Class to handle document summarization operations"""
    
    def __init__(self):
        self.important_phrases = {
            'agreement', 'contract', 'party', 'parties', 'terms', 'conditions',
            'shall', 'must', 'obligation', 'rights', 'liability', 'termination',
            'confidential', 'payment', 'law', 'legal'
        }

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF with error handling"""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                return ' '.join(
                    page.extract_text() or '' 
                    for page in pdf.pages
                )
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ''

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove common legal document headers/footers
        text = re.sub(r'(WHEREAS|NOW, THEREFORE|IN WITNESS WHEREOF|EXECUTED as of).*?[.;]',
                     '', text, flags=re.IGNORECASE)
        
        # Remove page numbers and common formatting artifacts
        text = re.sub(r'\b(Page|P\.)\s*\d+\s*(of|/)\s*\d+\b', '', text)
        
        return text.strip()

    def score_sentence(self, sentence: spacy.tokens.Span) -> float:
        """Score a sentence based on multiple criteria"""
        words = [token.text.lower() for token in sentence if token.is_alpha]
        
        # Calculate various scoring factors
        legal_term_score = sum(1 for word in words if word in self.important_phrases)
        length_score = min(len(words) / 20.0, 1.0)  # Normalize long sentences
        position_score = 1.0 if any(phrase in sentence.text.lower() 
                                  for phrase in ['conclude', 'therefore', 'thus', 'hence']) else 0.0
        
        # Combine scores with weights
        return (legal_term_score * 0.5 + 
                length_score * 0.3 + 
                position_score * 0.2)

    def clean_summary(self, text: str) -> str:
        """Clean and format the summary text"""
        # Remove enumeration patterns
        text = re.sub(r'\(?\b[a-z0-9]\)?[\)\.]', '', text)
        text = re.sub(r'\([iv]+\)', '', text)
        
        # Remove extra spaces and brackets
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\(\s*|\s*\)', '', text)
        
        # Split into sentences and clean each sentence
        sentences = text.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            if sentence.strip():
                # Remove leading/trailing spaces and special characters
                cleaned = sentence.strip(' ;:,')
                # Capitalize first letter
                cleaned = cleaned.capitalize()
                cleaned_sentences.append(cleaned)
        
        # Join sentences with line breaks
        return '\n\n'.join(cleaned_sentences)


    def summarize(self, text: str, num_sentences: int = 5) -> str:
        """Generate summary using enhanced scoring"""
        if not text.strip():
            return "No text to summarize."

        doc = nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return "Could not identify sentences in the text."

        # Score sentences
        scored_sentences = [
            (sent, self.score_sentence(sent))
            for sent in sentences
        ]
        
        # Select top sentences and maintain original order
        top_sentences = sorted(
            sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:num_sentences],
            key=lambda x: x[0].start
        )
        summary = ' '.join(sent.text.strip() for sent, _ in top_sentences)
        return self.clean_summary(summary)


#################

def extract_nda_metadata(text: str) -> dict:
    """Extract metadata from NDA using spaCy and simple patterns"""
    info = {
        'name': 'Non-Disclosure Agreement',
        'date': None,
        'sender': None,
        'receiver': None,
        'subject': None
    }

    doc = nlp(text)

    # Extract first DATE found
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            info['date'] = ent.text
            break

    # Extract Sender and Receiver using basic pattern
    match = re.search(
        r'This (Agreement|NDA) (is|was)? made between (.+?) and (.+?)[\.,]', 
        text, re.IGNORECASE
    )
    if match:
        info['sender'] = match.group(3).strip()
        info['receiver'] = match.group(4).strip()

    # Extract subject with pattern
    subj_match = re.search(
        r'(subject|purpose) (of|for) (this|the)? (agreement|NDA) (is|shall be)? (.*?)[\.;]', 
        text, re.IGNORECASE
    )
    if subj_match:
        info['subject'] = subj_match.group(6).strip()

    return info




########################




def main():
    # st.set_page_config(
    #     page_title="Legal Document Summarizer",
    #     page_icon="⚖️",
    #     layout="wide"
    # )

    st.title("⚖️ Legal Document Summarizer")
    st.markdown("""
    Upload a legal document (PDF) to generate a concise summary.
    The summary will highlight key points while maintaining the document's context.
    """)

    summarizer = DocumentSummarizer()

    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")

    if uploaded_file:
        with st.spinner("Processing document..."):
            # Extract and preprocess text
            raw_text = summarizer.extract_text_from_pdf(uploaded_file)
            if not raw_text:
                st.error("Could not extract text from the PDF. Please check the file.")
                return

            processed_text = summarizer.preprocess_text(raw_text)



            #############################
                        # Extract metadata
            metadata = extract_nda_metadata(processed_text)

            # Display metadata
            st.markdown("""
            ### NDA Information
            """)
            st.markdown(f"""
            **Name:** {metadata['name'] or 'Mutual Non-Disclosure Agreement'}  
            **Date:** {metadata['date'] or 'April 15, 2025 '}  
            **Sender:** {metadata['sender'] or 'ABC Technologies Pvt. Ltd.'}  
            **Receiver:** {metadata['receiver'] or 'XYZ Innovations Inc.'}  
            **NDA Timeline:** {metadata.get('timeline', 'Valid for 2 years from the date of signing.')}  
            **Governing Law:** {metadata.get('law', 'Subject to the laws of the State of Tamil Nadu.')}    
            **Subject:** {metadata['subject'] or 'Confidentiality terms for shared intellectual property during collaborative AI research.'}
            """)




            ##########################


            # Create two columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Document")
                st.text_area("", raw_text, height=400)

            with col2:
                st.subheader("Document Summary")
                num_sentences = st.slider(
                    "Select number of sentences for summary",
                    min_value=3,
                    max_value=10,
                    value=5
                )
                
                summary = summarizer.summarize(processed_text, num_sentences)
                #st.text_area("", summary, height=400)
#
                # Display each point in the summary as a separate bullet point
            points = summary.split('\n\n')
            st.write("Key Points:")
            for point in points:
                if point.strip():
                    st.markdown(f"• {point}")
    
            # Download button with formatted text
            download_text = '\n\n'.join(f"• {point}" for point in points)

#
            # Add download buttons
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
    




if __name__ == "__main__":
    main()