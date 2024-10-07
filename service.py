from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import json
import pandas as pd

# Load a pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the dataset
# dataset = load_dataset('wikisql')

output_dir = "./models1"

with open("train.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f.readlines()]


def preprocess_sql(sql_dict):
    """
    Convert the sql dictionary into a SQL query string.
    Example input: {"sel": 5, "conds": [[3, 0, "SOUTH AUSTRALIA"]], "agg": 0}
    """
    # Placeholder for column names and aggregation functions
    column_names = ["id", "name", "age", "location", "notes"]  # Update this as needed
    agg_funcs = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]  # Adjust based on your use case

    # Check and handle the 'sel' value
    sel_index = sql_dict.get("sel", -1)
    if 0 <= sel_index < len(column_names):
        selected_col = column_names[sel_index]
    else:
        selected_col = "unknown_column"  # Fallback or handle error
        print(f"Warning: 'sel' index {sel_index} out of range. Using fallback column.")

    # Get aggregation function
    agg_index = sql_dict.get("agg", 0)
    agg = agg_funcs[agg_index] if 0 <= agg_index < len(agg_funcs) else ""

    # Build the SELECT part of the query
    sql_query = f"SELECT {agg}({selected_col})" if agg else f"SELECT {selected_col}"

    # Build the WHERE clause if conditions are present
    if sql_dict.get("conds"):
        conditions = []
        for cond in sql_dict["conds"]:
            col_idx, operator, value = cond
            if 0 <= col_idx < len(column_names):
                column = column_names[col_idx]
                conditions.append(f"{column} = '{value}'")
            else:
                print(f"Warning: 'conds' column index {col_idx} out of range.")
        where_clause = " WHERE " + " AND ".join(conditions)
        sql_query += where_clause

    return sql_query


def preprocess_function(example):
    inputs = example['question']
    targets = preprocess_sql(example['sql'])

    # Tokenize
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels

    return model_inputs

def preprocess_function_new(row):
    # Extract the question and schema from the CSV row
    question = row['question']
    schema = row['schema']

    # Prepare the input by concatenating the question and schema
    input_text = f"Question: {question} Schema: {schema}"

    # Tokenize the inputs and outputs
    inputs = tokenizer(input_text, max_length=256, truncation=True, padding='max_length')

    # Tokenize the SQL query (output)
    sql_query = row['query']
    labels = tokenizer(sql_query, max_length=256, truncation=True, padding='max_length')

    # Add the tokenized input and label (SQL query) to the dataset
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': labels['input_ids']}


df = pd.read_csv('/Users/rashmiranjanswain/PycharmProjects/sqlQueryGeneration/data/train.csv')
preprocessed_dataset = [preprocess_function_new(row) for _, row in df.iterrows()]
dataset = Dataset.from_pandas(pd.DataFrame(preprocessed_dataset))
# Apply preprocessing to each example
# tokenized_dataset = [preprocess_function(ex) for ex in dataset]

# Fine-tune the model

# Convert list to Dataset
dataset = Dataset.from_pandas(pd.DataFrame(preprocessed_dataset))

# Split dataset if needed
# For example, split into training and validation sets
dataset = dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
