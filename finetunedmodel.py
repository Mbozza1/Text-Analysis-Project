from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
# Load the pre-trained summarization model and tokenizer
model_name = 't5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def fine_tune():
  # Load and preprocess the dataset
  dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
  dataset = dataset.map(preprocess, batched=True)
  dataset.set_format(type="torch",
                     columns=["input_ids", "attention_mask", "labels"])

  # Set up training arguments
  training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=500,
    save_steps=500,
    evaluation_strategy="epoch",
    fp16=True,
    learning_rate=3e-5,
  )

  # Train the model
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
  )

  trainer.train()

  # Save the fine-tuned model
  trainer.save_model("fine_tuned_t5")


# Function to generate a summary using the fine-tuned model
def summarize_fine_tuned(text, fine_tuned_model, max_length=500):
  # Tokenize the input text, truncate and pad if necessary
  input_ids = tokenizer.encode(text,
                               return_tensors="pt",
                               max_length=512,
                               truncation=True)
  # Generate the summary using the fine-tuned model
  output_ids = fine_tuned_model.generate(input_ids,
                                         max_length=max_length,
                                         num_return_sequences=1,
                                         early_stopping=True)
  # Decode the summary tokens and remove special tokens
  summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return summary


# Function to preprocess the input and target text for fine-tuning
def preprocess(example):
  # Add a prefix to the input text for the T5 model
  input_text = "summarize: " + example["article"]
  # Get the target text (highlights)
  target_text = example["highlights"]
  # Tokenize the input text, truncate and pad if necessary
  input_tokenized = tokenizer(input_text,
                              max_length=512,
                              padding="max_length",
                              truncation=True)
  # Tokenize the target text, truncate and pad if necessary
  target_tokenized = tokenizer(target_text,
                               max_length=150,
                               padding="max_length",
                               truncation=True)
  # Return the preprocessed input and target tokens
  return {
    "input_ids": input_tokenized["input_ids"],
    "attention_mask": input_tokenized["attention_mask"],
    "labels": target_tokenized["input_ids"],
  }
