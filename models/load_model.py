from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
