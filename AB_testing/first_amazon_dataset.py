import kagglehub
import os
import torch
import shutil
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

dataset_dir = r"/dataset"
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

path = kagglehub.dataset_download("karkavelrajaj/amazon-sales-dataset")
model = SentenceTransformer('all-mpnet-base-v2')

for filename in os.listdir(path):
    full_file_name = os.path.join(path, filename)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dataset_dir)

df = pd.read_csv(fr"{dataset_dir}/amazon.csv")
df = df.dropna()
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['user_id'] = df['user_id'].str.split(',')
df['product_name'] = df['product_name'].fillna('')
df['review_content'] = df['review_content'].fillna('')
df['about_product'] = df['about_product'].fillna('')
df['category'] = df['category'].fillna('')
df['actual_price'] = df['actual_price'].fillna('')
df['discount_percentage'] = df['discount_percentage'].fillna('')
df['about_product'] = df['about_product'].str.replace('|', '', regex=False)
df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

df_exploded = df.explode('user_id')
df_exploded = df_exploded.drop_duplicates(subset=['product_name', 'about_product'])
df_exploded = df_exploded.reset_index(drop=True)
df_exploded['product_id'] = df_exploded.index
category_counts = df_exploded['category'].value_counts()
print(category_counts)

#data для contenta
df_exploded['combined_features'] = (
    df_exploded['product_name'].astype(str) + ' ' +
    df_exploded['review_content'].astype(str) + ' ' +
    df_exploded['about_product'].astype(str) + ' ' +
    df_exploded['category'].astype(str) + ' ' +
    df_exploded['actual_price'].astype(str) + ' ' +
    df_exploded['discount_percentage'].astype(str)
)

#анализ
group1 = df_exploded.groupby('category')['rating'].mean().sort_values(ascending=False)
top_20_categories = group1.head(15)
plt.figure(figsize=(25, 18))
top_20_categories.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Mean rating in top 20 category')
plt.xlabel('Category')
plt.ylabel('Mean rating')
plt.ylim(2, 5)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Mean_rating.png')
plt.show()

group2 = df_exploded.groupby('category')['discount_percentage'].mean().sort_values(ascending=False)
top_20_categories = group2.head(15)
plt.figure(figsize=(25, 18))
top_20_categories.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Discount percentage in top 20 category')
plt.xlabel('Category')
plt.ylabel('Discount percentage')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Discount_percentage.png')
plt.show()

