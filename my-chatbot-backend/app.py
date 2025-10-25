import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_community.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate 
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

print(f"Value of GEMINI_API_KEY from environment: '{GEMINI_API_KEY}'")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")
genai.configure(api_key=GEMINI_API_KEY)

# Using gemini-2.5-flash for main chat and analysis
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
analysis_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=GEMINI_API_KEY)

# A separate model instance for quick intent classification (low temperature for reliability)
intent_classifier_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)


# Global variables
qa_chain = None
wiki_retriever = None

# Custom Prompt Template updated to be STRICTLY CONCISE.
CUSTOM_QA_PROMPT = """
You are a highly selective, professional, and concise information retrieval bot.
Your task is to answer the user's question only using the provided context from Wikipedia.

Strictly adhere to these rules:
1.  **Strict Conciseness:** The final answer must be a maximum of THREE sentences long. Only state the most crucial information.
2.  **Organization:** If the query is vague and multiple distinct concepts are found (e.g., Planet and Musician for "Mars"), summarize the main idea of each.
3.  **No Extraneous Text:** Do not include introductory phrases or conclusions. Start directly with the organized answer.

Context:
{context}

Question: {question}

Strictly Concise and Organized Answer (MAX THREE SENTENCES):
"""

# System prompt for intent classification
CLASSIFIER_SYSTEM_PROMPT = SystemMessage(
    content="You are a system that classifies user queries. Respond with ONLY ONE word: 'FACTUAL' if the query requires external knowledge retrieval (e.g., questions about history, science, specific people, or concepts), or 'CONVERSATIONAL' if it is a simple greeting, short command, compliment, or small talk."
)


def initialize_rag_system():
    global qa_chain, wiki_retriever
    print("Initializing Wikipedia RAG system...")
    try:
        # Define the custom prompt template
        QA_PROMPT_TEMPLATE = PromptTemplate(
            template=CUSTOM_QA_PROMPT, 
            input_variables=["context", "question"]
        )

        # 1. Initialize the Wikipedia Retriever
        # MODIFIED: Changed top_k_results from 3 to 1 to prioritize the single, most popular Wikipedia article (like Google SEO).
        wiki_retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=4000)
        print("WikipediaRetriever initialized.")

        # 2. Initialize the RetrievalQA chain using the WikipediaRetriever and custom prompt
        # The 'chain_type_kwargs' injects the strict prompt instructions into the LLM chain.
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=wiki_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT_TEMPLATE} # Inject the custom prompt here
        )
        print("Wikipedia RAG QA chain initialized with custom prompt for conciseness.")
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        qa_chain = None
        wiki_retriever = None

with app.app_context():
    initialize_rag_system()

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Example sets for non-sensitive data (Kept for compatibility)
NON_SENSITIVE_LOCATIONS = {
    "paris", "london", "tokyo", "new york", "mumbai", "chennai"
}
NON_SENSITIVE_NAMES = {
    "john", "jane", "alex", "michael"
}


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
        
        classification_response = intent_classifier_model.invoke(
            [CLASSIFIER_SYSTEM_PROMPT, user_message]
        ).content.strip().upper()

        is_factual_query = (classification_response == "FACTUAL")
        print(f"Intent classified as: {classification_response}")
        
        bot_response_text = ""
        
        if is_factual_query:
            # --- 2a. FACTUAL QUERY: Use RAG (Wikipedia) ---
            if not qa_chain:
                return jsonify({"error": "Wikipedia RAG system not initialized. Cannot answer factual questions."}), 500

            rag_response = qa_chain.invoke({"query": user_message})
            bot_response_text = rag_response["result"]
            
        else:
            # --- 2b. CONVERSATIONAL QUERY: Use direct LLM call ---
            # Define a standard conversational prompt for simple queries
            conversational_prompt = SystemMessage(
                content="You are a helpful and brief conversational assistant. Respond naturally and concisely to greetings or simple small talk. Do not retrieve external information."
            )
            direct_response = chat_model.invoke(
                [conversational_prompt, user_message]
            )
            bot_response_text = direct_response.content.strip()

        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Failed to get response from AI model.", "details": str(e)}), 500


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

    # --- HALLUCINATION CHECK: Now uses Wikipedia for context ---
    if qa_chain and wiki_retriever:
        try:
            # 1. Retrieve the relevant Wikipedia documents based on the text_to_analyze
            retrieved_docs_for_analysis = wiki_retriever.invoke(text_to_analyze)
            context_for_verification = "\n".join([doc.page_content for doc in retrieved_docs_for_analysis])

            if not context_for_verification.strip():
                # ENHANCEMENT: Improved message for irrelevant/unverifiable content
                analysis_results["hallucination_info"]["detected"] = True
                analysis_results["hallucination_info"]["reason"] = "Unverifiable Factual Claim: The statement could not be verified against the Wikipedia knowledge base. This may be due to the information being too obscure, too recent, or a complete hallucination."
                analysis_results["hallucination_info"]["correction"] = "No external context was found to support this claim."
            else:
                # 2. Use the retrieved context to ask Gemini for verification
                verification_prompt = f"""
                Given the following factual context retrieved from Wikipedia:
                ---
                {context_for_verification}
                ---
                Evaluate the following statement: "{text_to_analyze}"
                Does this statement align with, contradict, or is it not mentioned in the provided context?
                If it contradicts or is not mentioned, explain why and provide the correct information from the context if available.
                Provide your answer in a structured format:
                Verification: [Supported/Contradicted/Not Mentioned]
                Reasoning: [Explanation]
                Correction/Refinement: [Corrected statement if applicable, or 'N/A']
                """
                
                print(f"Sending verification prompt to Gemini:\n{verification_prompt}")
                
                verification_response = analysis_model.invoke(verification_prompt)
                verification_text = verification_response.content

                verification_status = "Unknown"
                reasoning = "N/A"
                correction = "N/A"

                if "Verification: Supported" in verification_text:
                    verification_status = "Supported"
                    reasoning_match = re.search(r"Reasoning: (.+)", verification_text)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                elif "Verification: Contradicted" in verification_text:
                    verification_status = "Contradicted"
                    reasoning_match = re.search(r"Reasoning: (.+)", verification_text)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                    correction_match = re.search(r"Correction/Refinement: (.+)", verification_text)
                    if correction_match:
                        correction = correction_match.group(1).strip()
                elif "Verification: Not Mentioned" in verification_text:
                    verification_status = "Not Mentioned"
                    reasoning_match = re.search(r"Reasoning: (.+)", verification_text)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                else:
                    reasoning = "Gemini's verification response format was unexpected. See raw response in console."
                    print(f"Raw Gemini verification response: {verification_text}")


                analysis_results["hallucination_info"]["detected"] = (verification_status == "Contradicted" or verification_status == "Not Mentioned")
                analysis_results["hallucination_info"]["reason"] = f"Factual check: {verification_status}. {reasoning}"
                analysis_results["hallucination_info"]["correction"] = correction if correction != 'N/A' else (text_to_analyze if verification_status == "Supported" else "Cannot provide a definitive correction without more context.")

        except Exception as e:
            print(f"Error during hallucination check with Gemini: {e}")
            analysis_results["hallucination_info"]["reason"] = f"Error during factual verification: {str(e)}. Cannot definitively verify."
    else:
        analysis_results["hallucination_info"]["reason"] = "Wikipedia RAG system not initialized. Cannot perform factual verification."
    # --- END HALLUCINATION CHECK ---


    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
