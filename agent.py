import pandas as pd
import re
import json
from langchain_ollama.llms import OllamaLLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate


def create_agent(filename: str):   ## Read CSV File
    df = pd.read_csv(filename)
    return df


def summarize_data(df: pd.DataFrame, num_rows=10):   ## Summarise the dataset
    sample = df.head(num_rows).to_dict(orient="records")
    columns = df.columns.tolist()
    return f"Columns: {columns}\nSample: {sample}"



def query_agent(agent_df: pd.DataFrame, query: str):  ## Creating a agent
    data_context = summarize_data(agent_df)

    prompt_template = PromptTemplate.from_template("""
                                                   
IMPORTANT:
You are an AI data assistant helping with CSV-based analysis.
NEVER include explanations or natural language.
ONLY return a single valid JSON object in one of the following formats:

Use only the provided dataset to answer questions. Return a valid JSON object using one of these formats:

- For a table:
{{"table": {{"columns": ["col1", "col2"], "data": [["val1", "val2"], ...]}}}}

- For a bar chart:
{{"bar": {{"columns": ["label", "series1", ...], "data": [["label1", 10, 20], ...]}}}}

- For a line chart:
{{"line": {{"columns": ["label", "series1", ...], "data": [["label1", 5, 15], ...]}}}}
                                                
- For other than JSON answer or Plain answer should come under this format:
{{"answer": "Plain or Other than JSON answer should here"}}

- For plain answers:
{{"answer": {{"key": "value", ...}}}}

- If unsure:
{{"answer": "I do not know."}}

Output must be a valid JSON object.
No natural language or markdown outside of the JSON.

Data Context:
{data_context}

Query: {query}
""")

    llm = OllamaLLM(
        model="llama2:7b",
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    chain = prompt_template | llm | StrOutputParser()

    # Now use invoke() properly with a single dict
    raw_response = chain.invoke({
        "data_context": data_context,
        "query": query
    })


    if isinstance(raw_response, str):
        try:
            json_str = re.search(r"(\{.*\})", raw_response, re.DOTALL).group(1)
            return json.loads(json_str)
        except:
            return {"text" : raw_response}
    elif isinstance(raw_response, dict):
        return raw_response  # Already parsed!
    else:
        return {"error": "Unexpected response type", "raw": str(raw_response)}
    