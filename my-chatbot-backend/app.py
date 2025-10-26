import os
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai
# We will catch the generic 'Exception' in the retry logic for robustness.

# Using a standard Wikipedia Retriever for verification context retrieval only
from langchain_community.retrievers import WikipediaRetriever 
# NEW: Import PubMedRetriever for scientific verification
from langchain_community.retrievers import PubMedRetriever
# NEW: Import Google Search API Wrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

print(f"Current working directory: {os.getcwd()}")

dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    print(f".env file found at: {dotenv_path}")
    load_success = load_dotenv(dotenv_path)
    print(f"load_dotenv() success: {load_success}")
else:
    print(f".env file NOT found at: {dotenv_path}")
    print("Please ensure your .env file is in the same directory as app.py")


app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# NEW: Ensure Google Search API Key is loaded
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(f"Value of GEMINI_API_KEY from environment: '{GEMINI_API_KEY}'")
# Note: Google Search requires both GOOGLE_CSE_ID and GOOGLE_API_KEY 
if not (GOOGLE_CSE_ID and GOOGLE_API_KEY):
    print("WARNING: GOOGLE_CSE_ID and/or GOOGLE_API_KEY are not set. Google Search grounding will be unavailable for analysis.")

if not GEMINI_API_KEY:
    # This print statement helps debug if the key is missing in the environment
    print("WARNING: GEMINI_API_KEY is not set. API calls will likely fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)


# CRITICAL FIX: Ensure the API key is correctly passed to the ChatGoogleGenerativeAI models.
# Using gemini-2.5-flash for main chat and analysis
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
# We use a *higher* temperature for the correction model to allow better synthesis
correction_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, google_api_key=GEMINI_API_KEY)
# The analysis model remains low-temp for reliability in RAG processing
analysis_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=GEMINI_API_KEY)

# A separate model instance for quick intent classification (low temperature for reliability)
intent_classifier_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)


# Global variables
# Initialization for Wikipedia Retriever
try:
    # INCREASED top_k_results from 3 to 5 for better coverage
    wiki_retriever = WikipediaRetriever(top_k_results=5, doc_content_chars_max=8000) 
    print("WikipediaRetriever initialized for analysis.")
except Exception as e:
    print(f"Error initializing WikipediaRetriever: {e}")
    wiki_retriever = None

# NEW: Initialization for PubMed Retriever
try:
    pubmed_retriever = PubMedRetriever(top_k_results=3, doc_content_chars_max=8000) 
    print("PubMedRetriever initialized for scientific analysis.")
except Exception as e:
    print(f"Error initializing PubMedRetriever: {e}")
    pubmed_retriever = None

# NEW: Initialization for Google Search Retriever
try:
    if GOOGLE_CSE_ID and GOOGLE_API_KEY:
        # Note: LangChain uses environment variables for Google Search
        search_wrapper = GoogleSearchAPIWrapper(k=3) # Retrieve up to 3 results
        print("GoogleSearchAPIWrapper initialized for analysis.")
    else:
        search_wrapper = None
        print("Google Search is disabled due to missing GOOGLE_CSE_ID or GOOGLE_API_KEY.")
except Exception as e:
    print(f"Error initializing GoogleSearchAPIWrapper: {e}")
    search_wrapper = None


# System prompt for intent classification
CLASSIFIER_SYSTEM_PROMPT = SystemMessage(
    content="You are a system that classifies user queries. Respond with ONLY ONE word: 'FACTUAL' if the query requires external knowledge retrieval (e.g., questions about history, science, specific people, or concepts), or 'CONVERSATIONAL' if it is a simple greeting, short command, compliment, or small talk."
)


analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Example sets for non-sensitive data (Kept for compatibility)
NON_SENSITIVE_LOCATIONS = {
    "paris", "london", "tokyo", "new york", "mumbai", "chennai"
}
NON_SENSITIVE_NAMES = {
    "john", "jane", "alex", "michael"
}


# --- NEW FUNCTION FOR API RETRY LOGIC (EXPONENTIAL BACKOFF) ---
def invoke_with_retry(model, prompt, max_retries=3, tools=None):
    """Invokes the LangChain model with exponential backoff on errors."""
    for attempt in range(max_retries):
        try:
            if tools:
                return model.invoke(prompt, tools=tools)
            else:
                return model.invoke(prompt)
        except Exception as e:
            # Catch all exceptions (API errors, connection issues, invalid arguments) for robust retrying
            error_type = "API/Connection Error"
            print(f"{error_type} (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                # Re-raise on the last attempt
                raise
            # Exponential backoff: 2^attempt seconds
            wait_time = 2 ** attempt
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
# --- END NEW FUNCTION ---


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    try:
        # --- 1. INTENT CLASSIFICATION ---
        print(f"Classifying intent for: '{user_message}'")
        
        # Use retry logic for classification
        classification_response_raw = invoke_with_retry(
            intent_classifier_model, 
            [CLASSIFIER_SYSTEM_PROMPT, user_message]
        ).content
        
        classification_response = classification_response_raw.strip().upper()

        is_factual_query = (classification_response == "FACTUAL")
        print(f"Intent classified as: {classification_response}")
        
        bot_response_text = ""
        
        # --- NEW CHECK FOR SPECULATIVE/INTERNAL KNOWLEDGE ONLY MODE ---
        is_speculative_mode = "internal knowledge only" in user_message.lower()
        if is_speculative_mode:
            is_factual_query = True # Treat as factual, but use special prompt
            print("Detected 'internal knowledge only' mode.")
        # --- END NEW CHECK ---
        
        if is_factual_query:
            
            # --- Setup Prompt based on mode ---
            if is_speculative_mode:
                # Option A: Speculative / internal knowledge only
                system_instruction = "You are an assistant answering strictly from your own internal knowledge. Do NOT fetch anything externally. If you are unsure, provide your best plausible answer and label it as [SPECULATIVE]. Be professional and concise."
            else:
                # Default Factual Mode: Use direct LLM call with Google Search Grounding
                system_instruction = "You are a helpful, professional, and concise assistant. Use Google Search to answer factual questions based on the latest information available."
            
            # Construct the chat history/prompt including system instructions
            prompt_with_history = [
                SystemMessage(content=system_instruction),
                user_message
            ]

            # Use retry logic for main chat model. Rely on the model's 
            # inherent ability to use search tools for factual queries.
            direct_response = invoke_with_retry(
                chat_model,
                prompt_with_history,
            )
            
            bot_response_text = direct_response.content.strip()
            
        else:
            # --- CONVERSATIONAL QUERY: Use direct LLM call ---
            # Define a standard conversational prompt for simple queries
            conversational_prompt = SystemMessage(
                content="You are a helpful and brief conversational assistant. Respond naturally and concisely to greetings or simple small talk. Do not retrieve external information."
            )
            
            # Use retry logic for conversational response
            direct_response = invoke_with_retry(
                chat_model,
                [conversational_prompt, user_message]
            )
            
            bot_response_text = direct_response.content.strip()

        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        # Return a more descriptive error in the response
        return jsonify({"error": "Failed to get response from AI model. Check if GEMINI_API_KEY is valid and network connectivity.", "details": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text_to_analyze = data.get('text')
    original_user_query = data.get('user_query', '').lower() # Convert to lower case for easy matching

    if not text_to_analyze:
        return jsonify({"error": "No text provided for analysis"}), 400

    analysis_results = {
        "original_text": text_to_analyze,
        "hallucination_info": {
            "detected": False,
            "reason": "N/A",
            "location": "N/A",
            "correction": "N/A"
        },
        "pii_info": {
            "detected": False,
            "details": [],
            "refined_text": text_to_analyze
        }
    }

    # --- ENHANCEMENT: Check for PII request in the user query regardless of chatbot response ---
    if re.search(r"phone number|contact number", original_user_query):
        analysis_results["pii_info"]["detected"] = True
        analysis_results["hallucination_info"]["detected"] = True # Flag as violation
        analysis_results["hallucination_info"]["reason"] = "Policy Violation Detected: The original query explicitly requested sensitive Personally Identifiable Information (PII), such as a phone number. Generating or searching for such private data is strictly prohibited by our policy."
        analysis_results["hallucination_info"]["correction"] = "This query falls under the policy violation category and should be flagged regardless of the chatbot's response."
        analysis_results["pii_info"]["details"].append({"message": "Query requested Phone Number (Policy Violation)." })
        # We can return immediately since the policy check takes precedence
        return jsonify(analysis_results)
    # --- END ENHANCEMENT ---


    # --- PII DETECTION: Only Phone Numbers in Chatbot Response ---
    initial_pii_analysis = analyzer.analyze(text=text_to_analyze, language="en", score_threshold=0.6)

    final_pii_results = []
    for result in initial_pii_analysis:
        # Only check for phone numbers as PII, as requested
        if result.entity_type == "PHONE_NUMBER":
            final_pii_results.append(result)

    if final_pii_results:
        # PII Detected logic (Enhanced messaging)
        analysis_results["pii_info"]["detected"] = True
        
        # Set a specific message for PII reason
        analysis_results["hallucination_info"]["detected"] = True # Flag as violation
        analysis_results["hallucination_info"]["reason"] = "Privacy Violation Detected: The statement contains sensitive Personally Identifiable Information (PII) like phone numbers, which should not be generated or shared as per our privacy policy."
        analysis_results["hallucination_info"]["correction"] = "Sensitive content was removed to comply with privacy policies."

        pii_details = []
        for entity in final_pii_results:
            pii_details.append({
                "type": entity.entity_type,
                "value": text_to_analyze[entity.start:entity.end],
                "location": f"Chars {entity.start}-{entity.end}",
                "score": f"{entity.score:.2f}"
            })
        analysis_results["pii_info"]["details"] = pii_details

        anonymized_result = anonymizer.anonymize(text=text_to_analyze, analyzer_results=final_pii_results)
        analysis_results["pii_info"]["refined_text"] = anonymized_result.text

        # If PII is detected, we return immediately and skip the factual hallucination check
        return jsonify(analysis_results)
    else:
        analysis_results["pii_info"]["details"].append({"message": "No PII (only Phone Numbers checked) detected."})
    # --- END PII DETECTION ---

    # --- HALLUCINATION CHECK: Three-tiered verification for context ---
    
    # 1. Start RAG process
    search_query = original_user_query 
    context_parts = []
    
    # Fetch Google Search context FIRST (most robust, provides snippets)
    if search_wrapper:
        print("Fetching context from Google Search...")
        search_results = search_wrapper.results(search_query, 3) 
        
        found_wiki_snippet = False
        search_context = []
        for res in search_results:
            snippet = res.get('snippet', 'N/A')
            title = res.get('title', 'N/A')
            source = res.get('source', '')
            search_context.append(f"Source: {title} ({source})\nSnippet: {snippet}")
            
            if 'wikipedia.org' in source.lower():
                found_wiki_snippet = True
                
        if search_context:
            context_parts.append(f"--- GOOGLE SEARCH CONTEXT ---\n" + "\n".join(search_context))

    
    # Fetch Wikipedia context: ONLY RUN if Google didn't give us a direct Wikipedia snippet
    if wiki_retriever and not found_wiki_snippet: 
        print("Fetching context from Wikipedia (Fallback)...")
        try:
            retrieved_wiki_docs = wiki_retriever.invoke(search_query)
            wiki_context = "\n".join([doc.page_content for doc in retrieved_wiki_docs])
            if wiki_context:
                context_parts.append(f"--- WIKIPEDIA CONTEXT ---\n{wiki_context}")
        except Exception as e:
            print(f"Error during Wikipedia retrieval: {e}")

    # Fetch PubMed context
    if pubmed_retriever:
        print("Fetching context from PubMed...")
        try:
            retrieved_pubmed_docs = pubmed_retriever.invoke(search_query)
            pubmed_context = "\n".join([f"Title: {doc.metadata.get('Title', 'N/A')}\nAbstract: {doc.page_content}" for doc in retrieved_pubmed_docs])
            if pubmed_context:
                context_parts.append(f"--- PUBMED CONTEXT ---\n{pubmed_context}")
        except Exception as e:
            print(f"Error during PubMed retrieval: {e}")
            
    
    context_for_verification = "\n\n".join(context_parts)
    
    # --- 2. Verification Logic ---
    if not context_for_verification.strip():
        # Fallback 1: If RAG provides NO context, use the LLM's native Google Search tool for a definitive answer.
        print("RAG failed to find context. Invoking LLM Search Grounding for correction...")
        
        # System instruction to force the LLM to search for the corrected information
        correction_prompt = SystemMessage(
            content=f"The statement to be verified is: '{text_to_analyze}'. The RAG system found no context. Use Google Search to find the most accurate factual correction or replacement statement for this topic. Respond ONLY with the corrected statement."
        )
        
        # Use retry logic with the correction model and tools enabled
        try:
            native_search_response = invoke_with_retry(
                correction_model,
                [correction_prompt],
                tools=[{"google_search": {}}]
            ).content.strip()

            analysis_results["hallucination_info"]["detected"] = True
            analysis_results["hallucination_info"]["reason"] = "RAG context failure detected. Correction provided via internal LLM search tool for robustness."
            analysis_results["hallucination_info"]["correction"] = native_search_response
            
        except Exception as e:
             # If the LLM search tool also fails
            analysis_results["hallucination_info"]["detected"] = True
            analysis_results["hallucination_info"]["reason"] = "Verification system failure: Both RAG and LLM Search failed to retrieve context. Cannot definitively verify."
            analysis_results["hallucination_info"]["correction"] = "Final system error: Cannot provide a definitive correction."


    else:
        # Fallback 2: If RAG HAS context, proceed with standard verification
        verification_prompt = f"""
        Given the following factual context retrieved from multiple sources (Wikipedia, Google Search, PubMed):
        ---
        {context_for_verification}
        ---
        Evaluate the following statement: "{text_to_analyze}"
        Does this statement align with, contradict, or is it not mentioned in the provided context?
        If it contradicts or is not mentioned, explain why. If it is Contradicted, or Not Mentioned but likely false, provide the most plausible correction from the provided context or state that a correction cannot be given.
        Provide your answer in a structured format:
        Verification: [Supported/Contradicted/Not Mentioned]
        Reasoning: [Explanation]
        Correction/Refinement: [Corrected statement if applicable, or 'N/A']
        """
        
        print(f"Sending verification prompt to Gemini:\n{verification_prompt}")
        
        try:
            # Use retry logic for analysis/verification model
            verification_response = invoke_with_retry(analysis_model, verification_prompt)
            verification_text = verification_response.content

            verification_status = "Unknown"
            reasoning = "N/A"
            correction = "N/A"

            # Use re.DOTALL to ensure multiline match if Gemini's response spans multiple lines
            if "Verification: Supported" in verification_text:
                verification_status = "Supported"
            elif "Verification: Contradicted" in verification_text:
                verification_status = "Contradicted"
            elif "Verification: Not Mentioned" in verification_text:
                verification_status = "Not Mentioned"
            
            # Extract Reasoning (single line)
            reasoning_match = re.search(r"Reasoning: (.+)", verification_text)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            
            # Extract Correction/Refinement (potentially multi-line)
            correction_match = re.search(r"Correction/Refinement: (.+)", verification_text, re.DOTALL)
            if correction_match:
                correction = correction_match.group(1).strip()


            # Handle logic for setting final results
            analysis_results["hallucination_info"]["detected"] = (verification_status == "Contradicted" or verification_status == "Not Mentioned")
            
            if analysis_results["hallucination_info"]["detected"]:
                analysis_results["hallucination_info"]["reason"] = f"Factual check: {verification_status}. {reasoning}"
                
                # If Gemini provided a correction, use it.
                if correction != 'N/A':
                    analysis_results["hallucination_info"]["correction"] = correction
                
                # --- CRITICAL FIX: IF CONTRADICTED BUT NO CORRECTION, FORCE NATIVE SEARCH ---
                elif verification_status == "Contradicted":
                    print("Contradiction detected but no structured correction found. Forcing native LLM search correction.")
                    
                    # System instruction to force the LLM to search for the corrected information
                    correction_prompt_contradicted = SystemMessage(
                        content=f"The RAG system confirmed the statement to be verified ('{text_to_analyze}') is CONTRADICTED by the context. Find the single most accurate factual replacement or correction for this statement using Google Search. Respond ONLY with the corrected statement."
                    )
                    
                    try:
                        forced_correction = invoke_with_retry(
                            correction_model,
                            [correction_prompt_contradicted],
                            tools=[{"google_search": {}}]
                        ).content.strip()
                        analysis_results["hallucination_info"]["correction"] = forced_correction
                    except Exception:
                        analysis_results["hallucination_info"]["correction"] = "Cannot provide a definitive correction without more context."
                        
                # If status is Not Mentioned, assume the statement is correct but unverified by RAG (e.g., future events) and set correction to original text for demonstration.
                elif verification_status == "Not Mentioned":
                        analysis_results["hallucination_info"]["correction"] = text_to_analyze

            else: # Supported
                analysis_results["hallucination_info"]["reason"] = f"Factual check: {verification_status}. {reasoning}"
                analysis_results["hallucination_info"]["correction"] = text_to_analyze

        except Exception as e:
            print(f"Error during RAG verification with Gemini: {e}")
            analysis_results["hallucination_info"]["reason"] = f"Error during RAG verification: {str(e)}. Cannot definitively verify."
    # --- END HALLUCINATION CHECK ---


    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
