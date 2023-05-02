import wikipediaapi
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import finetunedmodel

model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def main():
  # Fetch Wikipedia article text
  article_title = "quantum computing"
  article_text = get_wikipedia_article(article_title)

  if article_text:
    print("Summary before fine-tuning:")
    article_summary = summarize(article_text)
    print(article_summary)
    print("\n")

  ## am leaving this section as commented because without sufficient CPU capacity the code fails, however with sufficient CPU capacity it works
  # # Fine-tune the model
  # finetunedmodel.fine_tune()

  # # # Load the fine-tuned model
  # fine_tuned_model = T5ForConditionalGeneration.from_pretrained(
  #   "fine_tuned_t5")

  # if article_text:
  #   print("Summary after fine-tuning:")
  #   article_summary = finetunedmodel.summarize_fine_tuned(
  #     article_text, fine_tuned_model)
  #   print(article_summary)


# Function to generate a summary using the pre-trained model
def summarize(text, max_length=500):
  # Tokenize the input text, truncate and pad if necessary
  input_ids = tokenizer.encode(text,
                               return_tensors="pt",
                               max_length=512,
                               truncation=True)
  # Generate the summary using the pre-trained model
  output_ids = model.generate(input_ids,
                              max_length=max_length,
                              num_return_sequences=1,
                              early_stopping=True)
  # Decode the summary tokens and remove special tokens
  summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return summary


# Function to fetch Wikipedia article text
def get_wikipedia_article(title, language='en'):
  # Instantiate a Wikipedia object with the specified language
  wiki = wikipediaapi.Wikipedia(language)
  # Fetch the Wikipedia page for the given title
  page = wiki.page(title)

  # Check if the page exists and return None if it doesn't
  if not page.exists():
    print(f"The page '{title}' does not exist.")
    return None

  # Return the text content of the fetched Wikipedia page
  return page.text


if __name__ == "__main__":
  main()
