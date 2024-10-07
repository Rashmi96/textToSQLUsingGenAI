from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the app
app = FastAPI()

# Load the trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("/Users/rashmiranjanswain/PycharmProjects/sqlQueryGeneration/src/models1")
model = T5ForConditionalGeneration.from_pretrained("/Users/rashmiranjanswain/PycharmProjects/sqlQueryGeneration/src/models1/checkpoint-4632")


# Define the request body schema
class QueryRequest(BaseModel):
    question: str
    table_schema: str


# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to the SQL generation API"}


# SQL generation route
@app.post("/generate-sql/")
def generate_sql(query: QueryRequest):
    input_text = f"Question: {query.question} Schema: {query.table_schema}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"generated_sql": decoded_output}
