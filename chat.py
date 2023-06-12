import random
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/Shubu0103/shubhz-ai-folio-chat-model-base"
headers = {"Authorization": "Bearer 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"}

# from transformers import pipeline
# from transformers import AutoModelWithLMHead, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("flan-alpaca-base")
# model = AutoModelWithLMHead.from_pretrained("flan-alpaca-base")
# import torch
# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Is cuda available:", torch.cuda.is_available())
# model = model.to(device)
#print(model)

#prompt = "Write an email about an alpaca that likes flan"



# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

bot_name = "Shubham"

def get_response(msg):
    # model = pipeline(model="flan-alpaca-xl")
    #model(prompt, max_length=128, do_sample=True)
    print(msg)
    input_text = "question: %s " % (msg)
    response = requests.post(API_URL, headers=headers, json=input_text)
    print(response.text)
    output = [i['generated_text'] for i in json.loads(response.text)]
    if output:
        return output
    else:
        return "Model still loading..."

    # features = tokenizer([input_text], return_tensors='pt')
    # out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
    # if tokenizer.decode(out[0]):
    #     return tokenizer.decode(out[0])

    # sentence = tokenize(msg)
    # X = bag_of_words(sentence, all_words)
    # X = X.reshape(1, X    .shape[0])
    # X = torch.from_numpy(X).to(device)

    # output = model(X)
    # _, predicted = torch.max(output, dim=1)

    # tag = tags[predicted.item()]

    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]
    # if prob.item() > 0.75:
    #     for intent in intents['intents']:
    #         if tag == intent["tag"]:
    #             return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response({
            "inputs":sentence,
            "options":{"wait_for_model":True}

        })
        print(resp)

