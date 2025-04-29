import os
from dotenv import load_dotenv
from llm_wrapper import get_openai_llm, get_openai_embeddings, generate_answer, read_context_from_file
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

load_dotenv()

question = input("質問を入力してください: ")
context = read_context_from_file('data.txt')
ground_truth = "ここに正解を入力"
openai_llm = get_openai_llm()
openai_embeddings = get_openai_embeddings()
answer = generate_answer(openai_llm, question, context)

data = {
"question": [question],
"ground_truth": [ground_truth],
"answer": [answer],
"contexts": [[context]]
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

result = evaluate(
dataset,
metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
llm=openai_llm,
embeddings=openai_embeddings
)

print(f"生成された回答: {answer}")
print("評価結果:")
print(result)