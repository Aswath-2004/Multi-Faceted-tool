# Import necessary classes from the transformers library
from transformers import pipeline

# --- Chatbot Function ---
def run_chatbot_text2text(model_name="facebook/blenderbot-400M-distill"):
    """
    Runs a simple conversational chatbot using a text-to-text generation model.
    Manages conversation history by formatting it into the input prompt.

    Args:
        model_name (str): The name of the pre-trained model to use for the chatbot.
                          'facebook/blenderbot-400M-distill' is a good choice for dialogue.
    """
    try:
        # Initialize the text2text-generation pipeline.
        # This pipeline is suitable for models that generate text based on an input sequence,
        # including conversational models when history is formatted correctly.
        chatbot_pipeline = pipeline("text2text-generation", model=model_name)

        # Initialize conversation history as a list of strings
        # Each element will be a turn in the conversation (user or bot)
        conversation_history = []

        print(f"--- Simple Chatbot ({model_name}) ---")
        print("Type your message and press Enter. Type 'exit' to quit.")
        print("Bot: Hello! How can I help you today?")

        while True:
            user_input = input("You: ")

            if user_input.lower() == 'exit':
                break
            elif not user_input.strip():
                print("Bot: Please say something!")
                continue

            # Add user's input to history
            conversation_history.append(f"User: {user_input}")

            # Format the entire conversation history into a single string for the model
            # The model will generate the next part of the conversation (the bot's reply)
            # We join with a special separator (e.g., "\n")
            input_text = "\n".join(conversation_history) + "\nBot:"

            # Generate the bot's response
            # We set max_new_tokens to control the length of the bot's reply
            # and num_return_sequences=1 for a single best response.
            # We also set do_sample=True, top_k, and temperature for more varied responses.
            generated_output = chatbot_pipeline(
                input_text,
                max_new_tokens=50,  # Limit the length of the bot's new response
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                truncation=True # Truncate input if it gets too long for the model
            )

            # Extract the generated text from the output
            # The generated text will include the input_text, so we need to extract only the new bot's part.
            full_generated_text = generated_output[0]['generated_text']

            # Find the part that starts with "Bot:" and extract the response
            # This is a simple way to parse, more robust parsing might be needed for complex models
            if "Bot:" in full_generated_text:
                bot_response = full_generated_text.split("Bot:")[-1].strip()
            else:
                # Fallback if "Bot:" isn't found in the generated text (e.g., model generates something unexpected)
                bot_response = full_generated_text.strip()
                # You might want to refine this parsing based on actual model output behavior

            # Add bot's response to history
            conversation_history.append(f"Bot: {bot_response}")

            print(f"Bot: {bot_response}")

    except Exception as e:
        print(f"An error occurred during the chatbot session: {e}")
        print("Please ensure you have installed the necessary libraries (transformers, torch) and that your internet connection is stable for model download.")

# --- Main Execution ---
if __name__ == "__main__":
    # You can experiment with other text2text generation models if you wish.
    # For example: "google/flan-t5-small" (though it might require more specific prompt engineering for chat)
    run_chatbot_text2text(model_name="facebook/blenderbot-400M-distill")
 