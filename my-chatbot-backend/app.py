import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

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

chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=GEMINI_API_KEY)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
analysis_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GEMINI_API_KEY)

vectorstore = None
qa_chain = None

def initialize_rag_system():
    global vectorstore, qa_chain
    print("Initializing RAG system...")
    try:
        documents = []
        for filename in os.listdir("knowledge_base"):
            if filename.endswith(".txt"):
                filepath = os.path.join("knowledge_base", filename)
                loader = TextLoader(filepath)
                documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        vectorstore = Chroma.from_documents(chunks, embedding_model)
        print("ChromaDB vectorstore initialized.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        print("RAG QA chain initialized.")

    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        vectorstore = None
        qa_chain = None

with app.app_context():
    initialize_rag_system()

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Example sets for non-sensitive data
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

    if not qa_chain:
        return jsonify({"error": "RAG system not initialized. Cannot answer questions."}), 500

    try:
        rag_response = qa_chain.invoke({"query": user_message})
        bot_response_text = rag_response["result"]
        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": "Failed to get response from AI model.", "details": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text_to_analyze = data.get('text')
    original_user_query = data.get('user_query', '')

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

    # Use Presidio to get initial PII analysis
    initial_pii_analysis = analyzer.analyze(text=text_to_analyze, language="en", score_threshold=0.6)

    final_pii_results = []
    for result in initial_pii_analysis:
        # Only check for phone numbers as PII
        if result.entity_type == "PHONE_NUMBER":
            final_pii_results.append(result)

    if final_pii_results:
        analysis_results["pii_info"]["detected"] = True
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
    else:
        analysis_results["pii_info"]["details"].append({"message": "No PII detected by the enhanced analyzer."})

    if qa_chain:
        try:
            retrieved_docs_for_analysis = vectorstore.as_retriever().invoke(text_to_analyze)
            context_for_verification = "\n".join([doc.page_content for doc in retrieved_docs_for_analysis])

            if not context_for_verification.strip():
                analysis_results["hallucination_info"]["reason"] = "No relevant context found in knowledge base for verification. Cannot definitively verify factual accuracy."
            else:
                verification_prompt = f"""
                Given the following factual context:
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
        analysis_results["hallucination_info"]["reason"] = "RAG system not initialized. Cannot perform factual verification."


    return jsonify(analysis_results)

if __name__ == '__main__':
    app.run(debug=True)
