import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import re 
from collections import defaultdict

#Load data
df = pd.read_csv('job_dataset.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())  

df['Title'] = df['Title'].fillna('Unknown Role')
print(df.isnull().sum())

#Extract Domain from JobID
def extract_prefix(jid):
    jid = str(jid)
    if '-' in jid:
        return jid.split('-')[0].split('_')[0]  #Handles NET-F-001 → NET
    else:
        match = re.match(r'([A-Z]+)', jid)  #Handles AI001 → AI
        return match.group(1) if match else jid

df['Prefix'] = df['JobID'].apply(extract_prefix)

#Create mapping: Prefix → Most common (mode) Title as readable Domain name
domain_map = df.groupby('Prefix')['Title'].agg(lambda x: x.value_counts().index[0] if not x.empty else 'Unknown Role').to_dict()

#Apply mapping
df['Domain'] = df['Prefix'].map(domain_map)

#Parse YearsOfExperience to get Min_Exp (numeric minimum years)
def parse_min_exp(years_str):
    if pd.isna(years_str):
        return 0
    #Handle en-dash '–' and normal '-'
    cleaned = years_str.replace('–', '-')
    match = re.search(r'(\d+)', cleaned)
    return float(match.group(1)) if match else 0

df['Min_Exp'] = df['YearsOfExperience'].apply(parse_min_exp)

#Count number of skills and keywords
def count_items(item_str):
    if pd.isna(item_str):
        return 0
    items = [i.strip() for i in str(item_str).split(';') if i.strip()]
    return len(items)

df['Skills_Count'] = df['Skills'].apply(count_items)
df['Keywords_Count'] = df['Keywords'].apply(count_items)

#Clean skills for analysis (strip whitespace)
def clean_skills(skill_str):
    if pd.isna(skill_str):
        return []
    return [s.strip() for s in skill_str.split(';') if s.strip()]

df['Skills_List'] = df['Skills'].apply(clean_skills)

#Get top 20 domains by job count (adjust number if you want more/less)
top_domains = df['Domain'].value_counts().head(20).index
df_top = df[df['Domain'].isin(top_domains)]  

#Set style for all plots
#sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.figsize'] = (12, 8)

#1. Bar chart: Job count by Domain 
plt.figure(figsize=(12, 10))  
domain_counts = df_top['Domain'].value_counts().sort_values()  
sns.barplot(x=domain_counts.values, y=domain_counts.index, palette='viridis')
plt.title('1. Job Count by Domain (Top 20)')
plt.xlabel('Number of Jobs')
plt.ylabel('Domain')
plt.tight_layout()
plt.show()

#2. Pie chart: Proportion of ExperienceLevel 
plt.figure(figsize=(10, 10))  #Larger figure for clarity
exp_level_series = df['ExperienceLevel'].value_counts(normalize=True)
threshold = 0.03  #3% - change to 0.02 for stricter grouping
small_categories = exp_level_series[exp_level_series < threshold]
other_proportion = small_categories.sum()
exp_level_series = exp_level_series[exp_level_series >= threshold]
if other_proportion > 0:
    exp_level_series['Other'] = other_proportion
exp_level_series = exp_level_series.sort_values(ascending=False)

wedges, texts, autotexts = plt.pie(
    exp_level_series.values,
    labels=exp_level_series.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel', len(exp_level_series)),
    pctdistance=0.85,    
    labeldistance=1.05   
)

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

plt.title('2. Proportion of Experience Levels (Small Categories Grouped as "Other")', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()

#3. Grouped Bar chart: Job count by ExperienceLevel within each Domain 
plt.figure(figsize=(14, 8))
sns.countplot(data=df_top, x='Domain', hue='ExperienceLevel', palette='deep')
plt.title('3. Job Count by Experience Level within Each Domain (Top 20)')
plt.xlabel('Domain')
plt.ylabel('Count')
plt.xticks(rotation=90) 
plt.legend(title='Experience Level')
plt.tight_layout()
plt.show()

#4. Histogram: Distribution of Min_Exp
plt.figure()
sns.histplot(df['Min_Exp'], bins=15, kde=True, color='skyblue')
plt.title('4. Distribution of Minimum Experience Required (Years)')
plt.xlabel('Minimum Years of Experience')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#5. Box plot: Min_Exp by Domain 
plt.figure(figsize=(12, 10))
sns.boxplot(data=df_top.sort_values('Min_Exp'), y='Domain', x='Min_Exp', palette='Set2')  
plt.title('5. Minimum Experience Required by Domain (Top 20)')
plt.xlabel('Min Years of Experience')
plt.ylabel('Domain')
plt.tight_layout()
plt.show()

#6. Bar chart: Top 20 most frequent skills (overall) 
all_skills = [skill for sublist in df['Skills_List'] for skill in sublist]
skill_counts = pd.Series(all_skills).value_counts().head(20)

plt.figure()
sns.barplot(x=skill_counts.values, y=skill_counts.index, palette='mako')
plt.title('6. Top 20 Most Frequent Skills (Overall)')
plt.xlabel('Frequency')
plt.ylabel('Skill')
plt.tight_layout()
plt.show()

#7. Word cloud: All Skills combined
all_skills_text = ' '.join(all_skills)
wordcloud_skills = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_skills_text)

plt.figure()
plt.imshow(wordcloud_skills, interpolation='bilinear')
plt.axis('off')
plt.title('7. Word Cloud - All Skills')
plt.tight_layout()
plt.show()

#8. Word cloud: All Keywords combined 
all_keywords = []
for kw in df['Keywords'].dropna():
    all_keywords.extend([k.strip() for k in kw.split(';') if k.strip()])
keywords_text = ' '.join(all_keywords)
wordcloud_keywords = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(keywords_text)

plt.figure()
plt.imshow(wordcloud_keywords, interpolation='bilinear')
plt.axis('off')
plt.title('8. Word Cloud - All Keywords')
plt.tight_layout()
plt.show()

#9. Bar chart: Top 15 skills by Domain
plt.figure(figsize=(14, 8))
skills_by_domain = df_top.explode('Skills_List').groupby(['Domain', 'Skills_List']).size().unstack(fill_value=0)
top_skills = pd.Series(all_skills).value_counts().head(15).index
skills_by_domain = skills_by_domain[top_skills.intersection(skills_by_domain.columns)]

skills_by_domain.plot(kind='bar', stacked=False, colormap='tab20')
plt.title('9. Top 15 Skills Frequency by Domain (Top 20 Domains)')
plt.xlabel('Domain')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.legend(title='Skill', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#10. Bar chart: Average Skills_Count by Domain
plt.figure(figsize=(12, 10))
avg_skills_by_domain = df_top.groupby('Domain')['Skills_Count'].mean().sort_values()
sns.barplot(x=avg_skills_by_domain.values, y=avg_skills_by_domain.index, palette='coolwarm')
plt.title('10. Average Number of Required Skills by Domain (Top 20)')
plt.xlabel('Average Skills Count')
plt.ylabel('Domain')
plt.tight_layout()
plt.show()

#11. Bar chart: Average Keywords_Count by ExperienceLevel
plt.figure()
avg_keywords_by_exp = df.groupby('ExperienceLevel')['Keywords_Count'].mean().sort_values(ascending=False)
sns.barplot(x=avg_keywords_by_exp.index, y=avg_keywords_by_exp.values, palette='Spectral')
plt.title('11. Average Number of Keywords by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Average Keywords Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#12. Horizontal bar chart: Top 10 most common job Titles
title_counts = df['Title'].value_counts().head(10)
plt.figure()
sns.barplot(x=title_counts.values, y=title_counts.index, palette='husl')
plt.title('12. Top 10 Most Common Job Titles')
plt.xlabel('Frequency')
plt.ylabel('Job Title')
plt.tight_layout()
plt.show()

#13. Stacked bar: ExperienceLevel distribution within each Domain 
plt.figure(figsize=(14, 8))
pd.crosstab(df_top['Domain'], df['ExperienceLevel']).plot(kind='bar', stacked=True, colormap='Paired')
plt.title('13. Stacked Experience Level Distribution by Domain (Top 20)')
plt.xlabel('Domain')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Experience Level', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

#14. Violin plot: Skills_Count distribution by ExperienceLevel
plt.figure()
sns.violinplot(data=df, x='ExperienceLevel', y='Skills_Count', palette='muted')
plt.title('14. Distribution of Skills Count by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Number of Skills')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#15. Heatmap: Skill co-occurrence (top 20 skills)
top_20_skills = pd.Series(all_skills).value_counts().head(20).index.tolist()
co_occurrence = pd.DataFrame(0, index=top_20_skills, columns=top_20_skills, dtype=int)

for skills in df['Skills_List']:
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            skill1, skill2 = skills[i], skills[j]
            if skill1 in top_20_skills and skill2 in top_20_skills:
                co_occurrence.loc[skill1, skill2] += 1
                co_occurrence.loc[skill2, skill1] += 1

plt.figure(figsize=(14, 10))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlOrRd', linewidths=.5)
plt.title('15. Skill Co-occurrence Heatmap (Top 20 Skills)')
plt.tight_layout()
plt.show()
