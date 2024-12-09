from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Define model name or path to your pre-trained model
model_name = "dhruvgupta/better_ai_finetuned_llama"  # Replace with the actual model name or path

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"from_agent": "querry", "value_msg": '''Optimize the given code: def validate_email(self, email):
    if email.data != current_user.email:
        test_condition = User.query.filter_by(email=email.data).first()
        if test_condition:
            raise ValidationError('Email already exists in database.')'''}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128, use_cache = True)

print(_)