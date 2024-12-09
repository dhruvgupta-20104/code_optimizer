import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Function to process code
def process_code(code_snippet, model, tokenizer):
    messages = [{"from_agent": "querry", "value_msg": f"Optimize the given code: {code_snippet}"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    )
    text_streamer = TextStreamer(tokenizer)
    output = model.generate(input_ids=inputs['input_ids'], streamer=text_streamer, max_new_tokens=128, use_cache=True)
    return output


# Streamlit App
def main():
    # Define model name or path to your pre-trained model
    model_name = "dhruvgupta/better_ai_finetuned_llama"  # Replace with the actual model name or path

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model and specify the device map for CPU
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    st.title("Code Optimization Tool")

    st.write("Provide a code snippet below, and the app will optimize it and give suggestions.")

    # Text area for user input
    user_code = st.text_area("Enter your code snippet here:", height=300)

    # Button to process the code
    if st.button("Optimize Code"):
        if user_code.strip():
            # Process the code using the custom function
            output = process_code(user_code, model, tokenizer)

            # Display results
            st.subheader("Optimized Code:")
            st.code(output, language='python')

            st.subheader("Suggestions:")
            st.text(output)
        else:
            st.error("Please enter a code snippet to optimize.")

if __name__ == "__main__":
    main()
