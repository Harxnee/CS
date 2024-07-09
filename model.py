from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
from optimum.intel import OVModelForCausalLM
import os
import ipywidgets as widgets
from IPython.display import display, clear_output

# Hugging Face login
def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face token (or press Enter to skip if already logged in): ")
    if token:
        login(token)
        print("Logged in to Hugging Face")
    else:
        print("Proceeding without explicit login. Ensure you're already authenticated if needed.")

# Perform login
hf_login()

# Load the model and tokenizer
model_id = "OjasPatil/intel-llama2-7b-test3"
model = OVModelForCausalLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create the pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Initialize conversation history
conversation = []

# Function to get response from the model
def custom_model_response(message):
    prompt = f"<s>[INST]{message}[/INST]</s>"
    result = pipe(prompt)
    return result[0]['generated_text'].split('[/INST]')[-1].strip()

# Function to handle user input and update conversation
def on_submit(b):
    user_message = input_box.value
    conversation.append(("User", user_message))

    # Get response from your custom model
    bot_response = custom_model_response(user_message)
    conversation.append(("Bot", bot_response))

    # Clear input field and update conversation display
    input_box.value = ""
    output_box.value = "\n".join([f"{sender}: {message}" for sender, message in conversation])

# Create widgets
input_box = widgets.Text(description="User:")
submit_button = widgets.Button(description="Submit")
output_box = widgets.Textarea(description="Conversation:", layout=widgets.Layout(width='100%', height='300px'))

submit_button.on_click(on_submit)

# Display widgets
display(input_box, submit_button, output_box)

print("Chatbot is ready. Type your message and click Submit.")