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

path = kagglehub.dataset_download("asaniczka/amazon-uk-products-dataset-2023")
model = SentenceTransformer('all-mpnet-base-v2')

for filename in os.listdir(path):
    full_file_name = os.path.join(path, filename)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dataset_dir)

df_uk = pd.read_csv(fr"{dataset_dir}/amz_uk_processed_data.csv")

df_uk = df_uk.dropna()
df_uk['stars'] = pd.to_numeric(df_uk['stars'], errors='coerce')
df_uk['asin'] = df_uk['asin'].fillna('')
df_uk['title'] = df_uk['title'].fillna('')
df_uk['imgUrl'] = df_uk['imgUrl'].fillna('')
df_uk['price'] = df_uk['price'].fillna('')
df_uk['productURL'] = df_uk['productURL'].fillna('')
df_uk['reviews'] = df_uk['reviews'].fillna('')
df_uk['isBestSeller'] = df_uk['isBestSeller'].fillna('')
df_uk['boughtInLastMonth'] = df_uk['boughtInLastMonth'].fillna('')
df_uk['categoryName'] = df_uk['categoryName'].fillna('')
print(df_uk['categoryName'].unique())

target_categories = ['Lighting', 'Smart Speakers', 'Cameras', 'Torches', 'Coffee & Espresso Machines',
                     'Car & Motorbike', 'Smartwatches', 'Binoculars, Telescopes & Optics', 'Clocks', 'GPS, Finders & Accessories',
                     'Hi-Fi Receivers & Separates', 'Telephones, VoIP & Accessories']

target_lower = [cat.lower() for cat in target_categories]
mask = df_uk['categoryName'].str.lower().apply(lambda x: any(x.startswith(prefix) for prefix in target_lower))

df_uk = df_uk[mask].copy()
MAX_PER_CATEGORY = 60

def take_first_n(group):
    return group.head(MAX_PER_CATEGORY)

df_uk = df_uk.groupby('categoryName', group_keys=False).apply(take_first_n)
df_uk.reset_index(drop=True, inplace=True)

df_uk = df_uk.drop_duplicates()
category_counts = df_uk['categoryName'].value_counts()
df_uk = df_uk.reset_index(drop=True)
print(category_counts)


df_uk['combined_features'] = (
    df_uk['title'].astype(str) + ' ' +
    df_uk['categoryName'].astype(str) + ' ' +
    df_uk['price'].astype(str) + ' ' +
    df_uk['reviews'].astype(str)
)

print(df_uk['categoryName'].unique())

group1 = df_uk.groupby('categoryName')['price'].mean().sort_values(ascending=False)
top_20_categories = group1.head(15)
plt.figure(figsize=(25, 18))
top_20_categories.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Mean price in top 20 category')
plt.xlabel('Category')
plt.ylabel('Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Price.png')
plt.show()

group2 = df_uk.groupby('categoryName')['reviews'].mean().sort_values(ascending=False)
top_20_categories = group2.head(15)
plt.figure(figsize=(25, 18))
top_20_categories.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Mean quantity reviews in top 20 category')
plt.xlabel('Category')
plt.ylabel('Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Reviews.png')
plt.show()