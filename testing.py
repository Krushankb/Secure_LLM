# import os
# import torch
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# # print(torch.backends.mps.is_available())

# from transformers import pipeline
# mps_device = "mps"
# # pipe = pipeline('zero-shot-classification', device = mps_device)
# # seq = "i love watching the office show"
# # labels = ['negative', 'positive']
# # print(pipe(seq, labels))

# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # checkpoint = "Salesforce/codegen-350M-mono"
# # model = AutoModelForCausalLM.from_pretrained(checkpoint)
# # tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # text = "def hello_world():"

# # completion = model.generate(**tokenizer(text, return_tensors="pt"))

# # print(tokenizer.decode(completion[0]))

# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_id = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device = mps_device, max_new_tokens=10)
# hf = HuggingFacePipeline(pipeline=pipe)

# from langchain.prompts import PromptTemplate

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)

# chain = prompt | hf

# question = "What is electroencephalography?"

# print(chain.invoke({"question": question}))


# # from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen25-7b-multi", trust_remote_code=True)
# # model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen25-7b-multi")
# # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device = mps_device)
# # llm = HuggingFacePipeline(pipeline=pipe)

import pandas as pd
import sqlite3
from sqlalchemy import text

import streamlit as st

st.set_page_config(layout="wide")

def main():
    st.title("Welcome to P.L.A.S.M.A!")
    st.info("𝗣rivate 𝗟LM 𝗔utomatic 𝗦QL 𝗠achine 𝗔gent")
    con = sqlite3.connect("Chinook.db")

    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        print(df['Name'])
        st.success("SUCCESS!")
    # df.to_sql("titanic", con, if_exists='append', index=False)

    # st.success(con.execute("SELECT Name FROM titanic").fetchall())

if __name__ == "__main__":
  main()