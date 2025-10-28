import os
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# NEW: Import 'requests' for URL validation
import requests
import phonenumbers
import dns.resolver
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider # Required for custom recognizer injection

# --- RAG/LLM Tools ---
import google.generativeai as genai
from langchain_community.utilities import GoogleSearchAPIWrapper
# FIX: Replacing PubMedRetriever with the more stable ArXivRetriever for scientific context
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate 
from langchain_core.messages import SystemMessage


print(f"Current working directory: {os.getcwd()}")

dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    print(f".env file found at: {dotenv_path}")
    load_success = load_dotenv(dotenv_path)
    print(f"load_dotenv() success: {load_success}")
else:
    print(f"I'm sorry, I cannot start the application. The .env file was NOT found at: {dotenv_path}")
    print("Please ensure your .env file is in the same directory as app.py")


app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"Value of GEMINI_API_KEY from environment: '{GEMINI_API_KEY}'")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. API calls will likely fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)


# --- RAG & LLM SETUP ---
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
analysis_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=GEMINI_API_KEY)
# FIX: Increased temperature to 0.4 to ensure the correction model synthesizes text even when RAG is weak.
correction_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=GEMINI_API_KEY) 
intent_classifier_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)


# --- RAG RETRIEVER SETUP (Multi-Source Context) ---
try:
    wiki_retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=8000) 
    
    # FIX: Initialize ArXiv Retriever instead of PubMed
    arxiv_retriever = ArxivRetriever(top_k_results=3, doc_content_chars_max=8000)
    print("ArXivRetriever initialized for scientific analysis.")

    google_search_wrapper = GoogleSearchAPIWrapper(k=5)
    print("Multi-source RAG retrievers initialized.")
except Exception as e:
    print(f"Error initializing RAG retrievers: {e}")
    wiki_retriever = None
    arxiv_retriever = None
    google_search_wrapper = None

# --- PII & NLP SETUP ---

# Custom provider configuration to ensure PhoneRecognizer is used aggressively
provider = NlpEngineProvider(nlp_configuration={'nlp_engine_name': 'spacy', 
                                                'models': [{'lang_code': 'en', 
                                                            'model_name': 'en_core_web_lg'}]})
nlp_engine = provider.create_engine()

# FIX 1: Initialize PhoneRecognizer without unsupported arguments
phone_recognizer = PhoneRecognizer(context=["phone", "number", "contact", "call"])

# FIX 2: Initialize AnalyzerEngine without 'recognizer_list' argument, then manually add recognizer.
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
analyzer.registry.add_recognizer(phone_recognizer)


anonymizer = AnonymizerEngine()

# System prompt for intent classification
CLASSIFIER_SYSTEM_PROMPT = SystemMessage(
    content="You are a system that classifies user queries. Respond with ONLY ONE word: 'FACTUAL' if the query requires external knowledge retrieval (e.g., questions about history, science, specific people, or concepts), or 'CONVERSATIONAL' if it is a simple greeting, short command, compliment, or small talk."
)

# --- CONFIG: Adjust PII base risk for non-critical entities ---
BASE_RISK_ADJUSTMENTS = {
    "DATE_TIME": -0.45, 
    "NRP": -0.35,       
}

# --- NEW: Public Data Type Exception List (Non-PII, Non-Redactable) ---
PUBLIC_DATA_EXCEPTIONS = ["DATE_TIME", "NRP", "LOCATION", "ADDRESS"]


# --- HELPER FUNCTIONS ---

def invoke_with_retry(model, prompt, max_retries=3, tools=None):
    """Invokes the LangChain model with exponential backoff on errors."""
    for attempt in range(max_retries):
        try:
            if tools:
                return model.invoke(prompt, tools=tools)
            else:
                return model.invoke(prompt)
        except Exception as e:
            error_type = "API/Connection Error"
            print(f"{error_type} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                raise
            wait_time = 2 ** attempt
            time.sleep(wait_time)


# NEW: Custom URL Validity Checker
def check_url_validity(url):
    """Performs a lightweight HEAD request to check if a URL returns a 2xx or 3xx status code."""
    try:
        # Use HEAD request for speed and less bandwidth consumption
        response = requests.head(url, timeout=5, allow_redirects=True)
        # 2xx codes are success, 3xx codes are redirects (still valid)
        if 200 <= response.status_code < 400:
            return "VALID"
        elif response.status_code >= 400:
             return f"INVALID (Status: {response.status_code})"
        else:
             return "INVALID (Unknown Status)"
    except requests.RequestException as e:
        return f"INVALID (Request Failed: {type(e).__name__})"

# Function to extract all links from the text (simpler regex than Presidio's URL detector)
def extract_links(text):
    # Regex to capture http/https links
    url_pattern = re.compile(r'https?:\/\/[^\s\/\.]+[^\s]+')
    return url_pattern.findall(text)


def check_wikipedia_for_public_figure(name):
    """
    Uses Google Search to check for a high-confidence Wikipedia link for a name, 
    first using the full name, then falling back to the last name.
    """
    if not google_search_wrapper:
        return False
    
    name_parts = name.split()
    search_terms = [name] # Try full name first
    
    if len(name_parts) > 1:
        # Add the last name as a reliable fallback search term (e.g., 'Dhoni')
        search_terms.append(name_parts[-1]) 
    
    for term in search_terms:
        search_query = f'"{term}" wikipedia'
        print(f"Checking Public Figure status for: {search_query}")
        
        try:
            results = google_search_wrapper.results(search_query, num_results=3)
            
            for result in results:
                # Look for a Wikipedia link that also contains the search term in the title
                if 'wikipedia.org' in result.get('link', '').lower():
                    if term.lower() in result.get('title', '').lower():
                        print(f"Public Figure Check: Found strong Wikipedia match for {name} via term '{term}'.")
                        return True
        except Exception as e:
            print(f"Error during Google Search public figure check: {e}")
            # Do not return False here; continue to the next search term
            
    return False


def verify_pii_entity(text, entity, analysis_model):
    """
    Performs external checks (MX, Phone, LLM context, Public Figure) on a detected PII entity
    and calculates a risk score (0-1).
    """
    ent_text = text[entity.start:entity.end]
    ent_type = entity.entity_type
    
    signals = {}

    # --- Pre-Verification Filter (Removes Noise) ---
    if len(ent_text.split()) == 1 and len(ent_text) < 4:
         return None 
    
    # --- 1. External Validity Checks ---
    if ent_type == "PHONE_NUMBER":
        try:
            # Check for structural validity and format plausibility
            pn = phonenumbers.parse(ent_text, "IN") # Assume Indian number format for robust checking
            signals["phone_is_valid"] = phonenumbers.is_valid_number(pn)
        except Exception:
            signals["phone_is_valid"] = False

    if ent_type == "EMAIL_ADDRESS":
        try:
            domain = ent_text.split("@")[-1]
            dns.resolver.resolve(domain, "MX")
            signals["domain_check_mx"] = True
        except Exception:
            signals["domain_check_mx"] = False

    if ent_type == "PERSON":
        signals["public_figure"] = check_wikipedia_for_public_figure(ent_text)

    # --- 2. Contextual LLM Confirmation ---
    try:
        context = text[max(0, entity.start-50):min(len(text), entity.end+50)]
        
        llm_prompt = f"""
        You are a privacy expert. Evaluate the following entity within its context.
        Entity Type: {ent_text}
        Entity Value: "{ent_text}"
        Context Snippet: "...{context}..."

        Answer the question: Does this entity, in this specific context, represent **Personally Identifiable Information (PII)** that could identify a private individual?
        Respond ONLY with 'YES' or 'NO' followed by a one-sentence reason.
        """
        
        llm_resp = invoke_with_retry(analysis_model, llm_prompt).content
        
        signals["llm_confirmed_pii"] = "yes" in llm_resp.lower()
        signals["llm_reason"] = llm_resp.strip()
    except Exception as e:
        print(f"LLM PII context check failed: {e}")
        signals["llm_confirmed_pii"] = False
        signals["llm_reason"] = "LLM check failed due to API error."

    # --- 3. Score Calculation ---
    score = entity.score 
    
    score += BASE_RISK_ADJUSTMENTS.get(ent_type, 0)

    if signals.get("phone_is_valid"): score += 0.15
    if signals.get("domain_check_mx"): score += 0.10
    if signals.get("llm_confirmed_pii"): score += 0.25 

    # FIX: Increased penalty to -0.25 to successfully push public figures into the MEDIUM category.
    if signals.get("public_figure") and ent_type == "PERSON": score -= 0.25 
    
    # Normalize to 0-1 range
    final_score = min(max(score, 0), 1)
    
    verdict = "HIGH" if final_score > 0.7 else ("MEDIUM" if final_score > 0.4 else "LOW")

    return {
        "entity": ent_text,
        "type": ent_type,
        "presidio_score": f"{entity.score:.2f}", 
        "risk_score": f"{final_score:.2f}",
        "verdict": verdict,
        "signals": signals,
        # Required for matching in the main analyze function
        "start_index": entity.start,
        "end_index": entity.end
    }

def custom_redact_pii(text, pii_details, bot_text_start_index):
    """
    Manually redacts PII based on verified details, avoiding Presidio's anonymize() method.
    """
    
    # Sort details by start index, descending, to avoid messing up indices of subsequent entities
    sorted_details = sorted(pii_details, key=lambda x: x['start_index'], reverse=True)
    
    modified_text = list(text) # Convert string to list of characters for mutable modification

    for detail in sorted_details:
        start_index = detail['start_index']
        end_index = detail['end_index']
        entity_type = detail['type']
        
        # Check if the entity is a public figure that should NOT be redacted
        is_public_figure = detail['signals'].get('public_figure') and detail['type'] == 'PERSON'
        
        # Check if the entity is high risk and should be masked
        is_high_risk = detail['verdict'] == 'HIGH'
        
        # NEW FIX: Check if the entity is a public data type (Date, NRP, Location, Address)
        is_public_data_type = entity_type in PUBLIC_DATA_EXCEPTIONS

        # Rule: Only redact if it's HIGH risk OR if it's NOT a public person AND NOT public data.
        if is_high_risk or (not is_public_figure and not is_public_data_type): 
            
            # Use the mask replacement for generic sensitive data
            replacement_tag = f"<{entity_type}>"
            
            # Replace the segment in the list
            modified_text[start_index:end_index] = list(replacement_tag)
            
    # Reconstruct the string and trim the separator and user query if present
    final_text = "".join(modified_text)
    
    return final_text[bot_text_start_index:].strip()


# --- FLASK ROUTES ---

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    try:
        # --- 1. SPECIAL PII INJECTION MODE FOR TESTING ---
        if "generate pii" in user_message.lower():
            bot_response_text = f"""
            PII Test Data Block:
            Full Name: Elias Vance (DOB: 05/15/1988)
            Email: elias.vance@examplecorp.net
            Phone Number: +1-555-867-5309
            Address: 742 Evergreen Terrace, Springfield, OR 97477
            SSN (Fake): 900-01-0001
            """
            return jsonify({"response": bot_response_text})

        # --- 2. INTENT CLASSIFICATION ---
        print(f"Classifying intent for: '{user_message}'")
        
        classification_response_raw = invoke_with_retry(
            intent_classifier_model, 
            [CLASSIFIER_SYSTEM_PROMPT, user_message]
        ).content
        
        classification_response = classification_response_raw.strip().upper()
        is_factual_query = (classification_response == "FACTUAL")
        print(f"Intent classified as: {classification_response}")
        
        # --- 3. CONSTRUCT CHAT RESPONSE (Gemini with Google Search Grounding) ---
        
        system_instruction = "You are a helpful, professional, and concise assistant. Use Google Search to answer factual questions based on the latest information available."
        
        prompt_with_history = [
            SystemMessage(content=system_instruction),
            user_message
        ]

        direct_response = invoke_with_retry(
            chat_model,
            prompt_with_history,
        )
        
        bot_response_text = direct_response.content.strip()

        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Failed to get response from AI model. Check configuration.", "details": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text_to_analyze = data.get('text')  # Bot's response
    original_user_query = data.get('user_query', '') 
    
    if not text_to_analyze:
        return jsonify({"error": "No text provided for analysis"}), 400

    analysis_results = {
        "original_text": text_to_analyze,
        "hallucination_info": {
            "detected": False,
            "reason": "N/A",
            "correction": "N/A"
        },
        "pii_info": {
            "detected": False,
            "details": [],
            "refined_text": text_to_analyze
        }
    }
    
    # --- 1. INTENT CLASSIFICATION (Rerunning on User Query for Analysis Flow) ---
    try:
        classification_response_raw = invoke_with_retry(
            intent_classifier_model, 
            [CLASSIFIER_SYSTEM_PROMPT, original_user_query]
        ).content
        classification_response = classification_response_raw.strip().upper()
        is_factual_query = (classification_response == "FACTUAL")
    except Exception as e:
        print(f"Intent classification failed during analysis: {e}. Assuming Factual.")
        is_factual_query = True

    # --- 2. PII DETECTION RUNS ON COMBINED TEXT (User Query + Bot Response) ---
    
    # Combined text is used for accurate PII detection across the conversation context
    SEPARATOR = " |::SEPARATOR::| " 
    if original_user_query and original_user_query not in text_to_analyze:
        pii_detection_text = f"{original_user_query}{SEPARATOR}{text_to_analyze}"
    else:
        pii_detection_text = text_to_analyze
        
    separator_index = pii_detection_text.find(SEPARATOR)
    bot_text_start_index_in_combined = separator_index + len(SEPARATOR) if separator_index != -1 else 0

    # PII Detection (using Presidio on combined text)
    initial_pii_analysis = analyzer.analyze(text=pii_detection_text, language="en", score_threshold=0.6)
    
    pii_types_to_verify = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "DATE_TIME", "CREDIT_CARD", "NRP", "SSN", "ADDRESS"]
    
    pii_entities_in_bot_response_space = []
    high_risk_detected = False

    for entity in initial_pii_analysis:
        # FIX: Aggressive reclassification of numeric strings to PHONE_NUMBER
        if entity.entity_type == "DATE_TIME":
            # Check if the entity looks more like a phone number than a date
            numeric_string = re.sub(r'[^0-9]', '', pii_detection_text[entity.start:entity.end])
            if len(numeric_string) >= 8 and (len(numeric_string) <= 12):
                # Reclassify as PHONE_NUMBER if it fits the digit length criteria
                entity.entity_type = "PHONE_NUMBER"
                # Note: The Presidio object is mutable here, so we only need to change entity_type
                # print(f"DEBUG: Reclassified {entity_text_full} from DATE_TIME to PHONE_NUMBER.")

        if entity.entity_type not in pii_types_to_verify:
            continue
        
        # Filter out short/fragmented entities before costly verification
        entity_text_full = pii_detection_text[entity.start:entity.end]
        if len(entity_text_full.split()) == 1 and len(entity_text_full) < 4:
            continue

        # PII Verification (LLM, MX, Phone Validity, Public Figure)
        verified_data = verify_pii_entity(pii_detection_text, entity, analysis_model)
        
        if verified_data is not None:
            pii_entities_in_bot_response_space.append(verified_data)

            if verified_data['verdict'] == 'HIGH':
                high_risk_detected = True

    
    analysis_results["pii_info"]["details"] = pii_entities_in_bot_response_space

    if analysis_results["pii_info"]["details"]:
        analysis_results["pii_info"]["detected"] = True

        # 3. PII Redaction: Using manual string manipulation (THE STABLE FIX)
        
        # Use the custom function to redact PII and clean up the separator
        refined_text = custom_redact_pii(
            pii_detection_text,
            pii_entities_in_bot_response_space,
            bot_text_start_index_in_combined
        )
        
        analysis_results["pii_info"]["refined_text"] = refined_text

        
        # 4. Set PII Hallucination/Policy Flag
        analysis_results["hallucination_info"]["detected"] = high_risk_detected
        
        # FIX 1: Set Hallucination reason ONLY based on HIGH risk, otherwise defer to factual check
        if high_risk_detected:
            analysis_results["hallucination_info"]["reason"] = "Policy Violation Detected: The statement contains high-risk Personally Identifiable Information (PII). Sharing private data is strictly prohibited by our policy."
            analysis_results["hallucination_info"]["correction"] = analysis_results["pii_info"]["refined_text"]
        else:
            analysis_results["hallucination_info"]["reason"] = "PII detected but risk is medium/low. Redaction was applied for sensitive types."

        if high_risk_detected:
            return jsonify(analysis_results)
    
    # --- 3. RAG-BASED HALLUCINATION CHECK (Only runs if Factual and no HIGH-RISK PII was found) ---
    if not is_factual_query:
        analysis_results["hallucination_info"]["detected"] = False
        analysis_results["hallucination_info"]["reason"] = "Skipping factual check: User query was classified as CONVERSATIONAL."
        analysis_results["hallucination_info"]["correction"] = text_to_analyze
        return jsonify(analysis_results)


    search_query = original_user_query
    
    if wiki_retriever or arxiv_retriever or google_search_wrapper:
        print(f"Fetching context for verification using query: '{search_query}'")
        
        context_for_verification = ""
        
        # --- URL VALIDATION CONTEXT ---
        links_in_text = extract_links(text_to_analyze)
        link_validation_results = ""
        
        if links_in_text:
            link_validation_results = "\n--- URL VALIDATION RESULTS ---"
            for link in links_in_text:
                validation_status = check_url_validity(link)
                link_validation_results += f"\nURL: {link}, Status: {validation_status}"
            
            context_for_verification += link_validation_results + "\n"
        # --- END URL VALIDATION CONTEXT ---


        # 1. Google Search Context (Best for real-time/niche)
        if google_search_wrapper:
            try:
                print("Fetching context from Google Search...")
                google_results = google_search_wrapper.results(search_query, num_results=5)
                google_context = "\n".join([f"Source: {r.get('title', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}" for r in google_results])
                context_for_verification += "--- GOOGLE SEARCH CONTEXT ---\n" + google_context + "\n"
            except Exception as e:
                print(f"Google Search retrieval failed: {e}")
                
        # 2. Wikipedia Context (Structured general knowledge)
        if wiki_retriever:
            try:
                print("Fetching context from Wikipedia...")
                wiki_docs = wiki_retriever.invoke(search_query)
                wiki_context = "\n".join([doc.page_content for doc in wiki_docs])
                context_for_verification += "--- WIKIPEDIA CONTEXT ---\n" + wiki_context + "\n"
            except Exception as e:
                print(f"Wikipedia retrieval failed: {e}")

        # 3. ArXiv Context (Scientific/Academic authority)
        if arxiv_retriever:
            try:
                print("Fetching context from ArXiv...")
                arxiv_docs = arxiv_retriever.invoke(search_query)
                arxiv_context = "\n".join([f"Title: {doc.metadata.get('Title', 'N/A')}\nAbstract: {doc.page_content}" for doc in arxiv_docs])
                context_for_verification += "--- ARXIV CONTEXT ---\n" + arxiv_context + "\n"
            except Exception as e:
                print(f"ArXiv retrieval failed: {e}")

        if not context_for_verification.strip():
            # If ALL retrievers fail to find context
            analysis_results["hallucination_info"]["detected"] = True
            analysis_results["hallucination_info"]["reason"] = "Unverifiable Factual Claim: No external context could be retrieved to verify this statement."
            analysis_results["hallucination_info"]["correction"] = "No external context was found to support this claim."
        else:
            # 4. Use LLM to Verify Statement against Context
            verification_prompt = f"""
            Given the following factual context retrieved from multiple sources (Wikipedia, Google Search, ArXiv) AND the following URL validation results:
            ---
            {context_for_verification}
            ---
            
            Carefully evaluate the following statement: "{text_to_analyze}"
            
            1. Factual Check: Does the statement align with, contradict, or is it not mentioned in the factual context (Wikipedia, Google, ArXiv)?
            2. Link Check: If the statement contains URLs, which ones were marked as 'INVALID' in the validation results? You must remove all invalid URLs and replace them with the validation status in parentheses (e.g., [Link to paper](INVALID LINK)).
            
            If the statement is Contradicted or Not Mentioned, explain why and provide the corrected version, ensuring all invalid links are removed.
            
            Provide your answer in a structured format:
            Verification: [Supported/Contradicted/Not Mentioned]
            Reasoning: [Explanation]
            Correction/Refinement: [Corrected statement with only VALID links, or 'N/A']
            """
            
            print(f"Sending verification prompt to Gemini...")
            
            try:
                verification_response = invoke_with_retry(analysis_model, verification_prompt)
                verification_text = verification_response.content
            except Exception as e:
                print(f"Error during Gemini verification step: {e}")
                verification_text = "Verification: Not Mentioned\nReasoning: LLM verification failed.\nCorrection/Refinement: N/A"

            # 5. Extract results from Gemini's structured response
            verification_status = "Unknown"
            reasoning = "LLM response parsing failed."
            correction = "N/A"

            status_match = re.search(r"Verification:\s*(.+)", verification_text)
            reasoning_match = re.search(r"Reasoning:\s*(.+)", verification_text)
            correction_match = re.search(r"Correction/Refinement:\s*(.+)", verification_text, re.DOTALL) # Use DOTALL for multi-line capture

            if status_match:
                verification_status = status_match.group(1).strip()
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            if correction_match:
                correction = correction_match.group(1).strip()

            analysis_results["hallucination_info"]["detected"] = (verification_status == "Contradicted" or verification_status == "Not Mentioned")
            
            # FIX 2: Hallucination Reason set only if PII HIGH risk detection was skipped
            if analysis_results["hallucination_info"]["detected"]:
                analysis_results["hallucination_info"]["reason"] = f"Factual check: {verification_status}. {reasoning}"
            else: # If Supported, just use the reasoning
                 analysis_results["hallucination_info"]["reason"] = f"Factual check: {verification_status}. {reasoning}"

            analysis_results["hallucination_info"]["correction"] = correction if correction != 'N/A' else (text_to_analyze if verification_status == "Supported" else "N/A")

            # 6. Fallback Search for Correction (If Contradicted but correction failed)
            if verification_status == "Contradicted" and (correction == "N/A" or correction == analysis_results["hallucination_info"]["correction"] == "N/A"):
                print("Contradiction detected, but correction failed. Initiating native search for correction...")
                try:
                    correction_prompt = [
                        SystemMessage(content="You are a meticulous fact-checker. Given that the user's previous statement was contradicted by evidence, your sole task is to perform a Google Search and provide the single, correct, factual statement that addresses the core topic. Do not include your reasoning or any fluff."),
                        f"Provide the single correct, factual statement for: {text_to_analyze}"
                    ]
                    
                    correction_response = invoke_with_retry(
                        correction_model,
                        correction_prompt,
                        tools=[{"google_search": {}}]
                    )
                    
                    final_correction = correction_response.content.strip()
                    analysis_results["hallucination_info"]["correction"] = final_correction
                except Exception as e:
                    print(f"Final native search correction failed: {e}")
                    analysis_results["hallucination_info"]["correction"] = "Cannot provide a definitive correction without more context."
            
            # Final check for Correction/Not Mentioned default
            if analysis_results["hallucination_info"]["correction"] == "N/A":
                 analysis_results["hallucination_info"]["correction"] = "Cannot provide a definitive correction without more context."

    else:
        analysis_results["hallucination_info"]["reason"] = "RAG system not fully initialized. Cannot perform factual verification."


    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
