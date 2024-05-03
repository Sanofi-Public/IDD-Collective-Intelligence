import matplotlib.pyplot as plt
import math
from pywaffle import Waffle
import umap
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from matplotlib.ticker import LogFormatter
from scipy import stats
import re
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy.stats import mode
import rdkit
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from collections import Counter
from sklearn.metrics import auc
import pywaffle
from tqdm import tqdm 
from matplotlib.colors import BoundaryNorm
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.ndimage
from rdkit.Chem import PandasTools
import matplotlib.patches as mpatches

def analyze_data(df):
	"""
	Analyzes data by computing the mean result and the most frequently given answer per slide.

	Parameters:
	- df: DataFrame containing the columns 'Slide_ID', 'Result', and 'Answer'.

	Returns:
	- DataFrame with the mean result and the most frequently given answer for each slide.
	"""
	grouped = df.groupby('Slide_ID')
	success_rate = grouped['Result'].mean()
	most_given_answer = grouped['Answer'].agg(lambda x: mode(x)[0][0])

	analysis_result = pd.DataFrame({
		'Success_Rate': success_rate,
		'Most_Given_Answer': most_given_answer
	}).reset_index()

	analysis_result['Slide_ID'] = analysis_result['Slide_ID'].apply(lambda x: str(int(x)))

	return analysis_result


def remove_consistent_chemists(df):
	"""
	Removes chemists from the dataset who have shown consistent answers and certitudes across all entries.

	Parameters:
	- df: DataFrame containing 'Chemist', 'Certitude', and 'Answer' columns.

	Returns:
	- DataFrame with inconsistent chemists.
	"""
	consistent_chemists = df.groupby('Chemist').filter(lambda x: x['Certitude'].nunique() == 1 and x['Answer'].nunique() == 1)
	chemists_to_remove = consistent_chemists['Chemist'].unique()
	return df[~df['Chemist'].isin(chemists_to_remove)]


def compute_most_frequent_combined_weight_Endpoint(df):
	"""
	Computes the most frequent answer for each endpoint and question, weighted by certitude and chemist level.

	Parameters:
	- df: DataFrame containing 'Endpoint', 'Question', 'Answer', 'Certitude', 'Chemist Level', and 'Correct_Answer'.

	Returns:
	- DataFrame with most frequent answers and their correctness for each question and endpoint.
	"""
	def weighted_combined_most_frequent(group):
		answers = group['Answer'].unique()
		max_weighted_answer = None
		max_weight = -np.inf

		for answer in answers:
			subset = group[group['Answer'] == answer]
			weight_sum = ((subset['Certitude'] + subset['Chemist Level']) / 10).sum()
			if weight_sum > max_weight:
				max_weight = weight_sum
				max_weighted_answer = answer

		return max_weighted_answer

	grouped = df.groupby(['Endpoint', 'Question']).apply(weighted_combined_most_frequent).reset_index(name='Most_Frequent_Answer')
	grouped = grouped.merge(df[['Question', 'Correct_Answer']].drop_duplicates(), on='Question', how='left')
	grouped['Most_Frequent_Correct'] = (grouped['Most_Frequent_Answer'] == grouped['Correct_Answer']).astype(int)

	return grouped


def compute_most_frequent_Endpoint(df, weighted=False):
	"""
	Computes the most frequent answer for each endpoint and question, optionally weighted by certitude.

	Parameters:
	- df: DataFrame containing 'Endpoint', 'Question', 'Answer', 'Certitude', and 'Correct_Answer'.
	- weighted: Boolean indicating whether to weight answers by certitude levels (default is False).

	Returns:
	- DataFrame with most frequent answers and their correctness for each question and endpoint.
	"""
	weights = {1: 1/5, 2: 2/5, 3: 3/5, 4: 4/5, 5: 1}

	def weighted_most_frequent(group):
		answers = group['Answer'].unique()
		max_weighted_answer = None
		max_weight = -np.inf

		for answer in answers:
			subset = group[group['Answer'] == answer]
			if weighted:
				weight_sum = subset['Certitude'].map(weights).sum()
			else:
				weight_sum = len(subset)
			if weight_sum > max_weight:
				max_weight = weight_sum
				max_weighted_answer = answer

		return max_weighted_answer

	grouped = df.groupby(['Endpoint', 'Question']).apply(weighted_most_frequent).reset_index(name='Most_Frequent_Answer')
	grouped = grouped.merge(df[['Question', 'Correct_Answer']].drop_duplicates(), on='Question', how='left')
	grouped['Most_Frequent_Correct'] = (grouped['Most_Frequent_Answer'] == grouped['Correct_Answer']).astype(int)

	return grouped






def compute_scores(df):
	score = df.groupby('Chemist')['Result'].mean().reset_index()
	score.columns = ['Chemist', 'Score']
	score = score.merge(df[['Chemist', 'Chemist Level', 'Certitude']].drop_duplicates(), on='Chemist')
	return score







def assign_chemist_group(level):
	if level < 3:
		return 1  # non-expert
	elif level >=3 and level <= 5:
		return 2  # expert
	else:
		return 3  # al







# Function to reclassify chemists based on success rate
def reclassify_chemist_by_sr(df):
	d_new = []
	score_l = df["Score"].tolist()
	score_k = df["Chemist Level"].tolist()

	for r in range(len(df["Score"].tolist())):
		if score_k[r]==6:
			d_new.append(3)
		else:
			if score_l[r] > 0.5:
				d_new.append(2)
			else:
				d_new.append(1)
	
	
	return d_new


def columns_to_dict(df, key_column, value_column):
	return df.set_index(key_column)[value_column].to_dict()




def transform_dataset_B_v2(df):
	df = df.replace({"✓": 1, "": 0})
	question_nums = sorted(set(int(col.split(":")[0][1:]) for col in df.columns if "Q" in col))
	Qs = []
	Answers = []
	Certitudes = []
	for q_num in question_nums:
		q_cols = [col for col in df.columns if f"Q{q_num}:" in col]
		answer_column = df[q_cols].idxmax(axis=1).fillna("None").astype(str).str.split(":").str[1].str.strip()
		if q_num == 1:
			Qs.append(answer_column)
		elif q_num %2:
			Answers.append(answer_column)
		else:
			Certitudes.append(answer_column)
	df["Chemist Level"] = Qs[0]
	for i, cert in enumerate(Answers, 1):
		df[f"Answer Q{i}"] = cert
	for i, cert in enumerate(Certitudes, 1):
		df[f"Certitude Q{i}"] = cert
	df["Chemist"] = ['Chemist_'+str(i+1) for i in range(len(df))]
	inside = []
	for c in df.columns.tolist():
		if ":" not in c:
			inside.append(c)
	df = df[inside]
	return df



def _merge_data_CI(CI_s1_path, CI_s2_path, CI_structures_path):

    df_A = pd.read_csv(CI_structures_path, sep=',')
    df_B = pd.read_csv(CI_s1_path, sep = ";")

    df_B_transformed = transform_dataset_B_v2(df_B.copy()) 

    # Melting the first dataframe
    df1_melted = df_B_transformed.melt(id_vars=['Chemist Level', 'Chemist'], value_vars=[col for col in df_B_transformed.columns if 'Answer Q' in col], var_name='Question', value_name='Answer')

    df1_melted['Slide_ID'] = df1_melted['Question'].str.extract('(\d+)').astype(float)

    merged_df = df1_melted.merge(df_A, on='Slide_ID', how='left')

    df1_melted_cert = df_B_transformed.melt(id_vars=['Chemist Level', 'Chemist'], value_vars=[col for col in df_B_transformed.columns if 'Certitude Q' in col], var_name='Question', value_name='Certitude')

    df1_melted_cert['Slide_ID'] = df1_melted_cert['Question'].str.extract('(\d+)').astype(float)

    merged_with_cert = merged_df.merge(df1_melted_cert[['Chemist', 'Chemist Level', 'Slide_ID', 'Certitude']], on=['Chemist', 'Chemist Level', 'Slide_ID'], how='left')

    merged_with_cert = merged_with_cert.dropna().rename(columns = {"Certitude":"Answer", "Answer":"Certitude"})
    merged_with_cert.to_csv('../data/CI_Answer_A.csv', index = None)

    df_B = pd.read_csv(CI_s2_path, sep = ";")
    df_B_transformed = transform_dataset_B_v2(df_B.copy()) 

    col = []
    for c in df_B_transformed.columns.tolist():

        if "Q" in c:
            if "Answer" in c:
                col.append("Answer Q"+ str(int(c.replace("Answer Q", ""))+37))
            if "Certitude" in c:
                col.append("Certitude Q"+ str(int(c.replace("Certitude Q", ""))+37))
        else:
            col.append(c)

    df_B_transformed.columns = col
    df1_melted = df_B_transformed.melt(id_vars=['Chemist Level', 'Chemist'], value_vars=[col for col in df_B_transformed.columns if 'Answer Q' in col], var_name='Question', value_name='Answer')

    df1_melted['Slide_ID'] = df1_melted['Question'].str.extract('(\d+)').astype(float)

    merged_df = df1_melted.merge(df_A, on='Slide_ID', how='left')

    df1_melted_cert = df_B_transformed.melt(id_vars=['Chemist Level', 'Chemist'], value_vars=[col for col in df_B_transformed.columns if 'Certitude Q' in col], var_name='Question', value_name='Certitude')

    df1_melted_cert['Slide_ID'] = df1_melted_cert['Question'].str.extract('(\d+)').astype(float)

    merged_with_cert = merged_df.merge(df1_melted_cert[['Chemist', 'Chemist Level', 'Slide_ID', 'Certitude']], on=['Chemist', 'Chemist Level', 'Slide_ID'], how='left')

    merged_with_cert = merged_with_cert.dropna().rename(columns = {"Certitude":"Answer", "Answer":"Certitude"})
    merged_with_cert.to_csv('../data/CI_Answer_B.csv', index = None)


    df_A = pd.read_csv('../data/CI_Answer_A.csv', sep=',')
    df_B = pd.read_csv('../data/CI_Answer_B.csv', sep=',')

    df_B = remove_consistent_chemists(df_B)
    df_A = remove_consistent_chemists(df_A)

    df_A.to_csv('../data/CI_Answer_A.csv', index = None)
    df_B.to_csv('../data/CI_Answer_B.csv', index = None)
    # Creates new column 
    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)

    combined_df = pd.concat([df_A, df_B])
    combined_df = combined_df[combined_df['Certitude'].isna()!=True]

    df_A = pd.read_csv('../data/CI_Answer_A.csv', sep=',')
    df_B = pd.read_csv('../data/CI_Answer_B.csv', sep=',')
    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
    combined_scores = pd.concat([compute_scores(df_A), compute_scores(df_B)], ignore_index=True)
    combined_scores_all = combined_scores.copy()
    combined_scores_all["Chemist Level"] = 6
    combined_scores = pd.concat([combined_scores_all, combined_scores])
    df_A_all = df_A.copy()
    df_A_all["Chemist Level"] = 6
    df_A = pd.concat([df_A_all, df_A])
    df_B_all = df_B.copy()
    df_B_all["Chemist Level"] = 6
    df_B = pd.concat([df_B_all, df_B])
    df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group)
    df_B["Chemist Group"] = df_B["Chemist Level"].apply(assign_chemist_group)
    combined_scores["Chemist Group"] = combined_scores["Chemist Level"].apply(assign_chemist_group)
    most_frequent_combined = pd.concat([compute_most_frequent(df_A, False), compute_most_frequent(df_B, False)])
    most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
    most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
    most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
    most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
    most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
    combined_scores = pd.concat([compute_scores(df_A), compute_scores(df_B)], ignore_index=True)
    combined_scores_all = combined_scores.copy()
    combined_scores_all["Chemist Level"] = 6
    combined_scores = pd.concat([combined_scores_all, combined_scores])
    df_A_all = df_A.copy()
    df_A_all["Chemist Level"] = 6
    df_A = pd.concat([df_A_all, df_A])
    df_B_all = df_B.copy()
    df_B_all["Chemist Level"] = 6
    df_B = pd.concat([df_B_all, df_B])
    df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group)
    df_B["Chemist Group"] = df_B["Chemist Level"].apply(assign_chemist_group)
    combined_scores["Chemist Group"] = combined_scores["Chemist Level"].apply(assign_chemist_group)
    most_frequent_combined = pd.concat([compute_most_frequent(df_A, False), compute_most_frequent(df_B, False)])
    most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
    most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
    most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
    most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
    most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
    return(combined_df)


def plot_aggregation_AM(ax, csv_file_paths, aggregation_methods, admet):
    """Plot aggregation of ADMET data using different methods.

    Args:
    ax (matplotlib.axis): The axis object to plot on.
    csv_file_paths (list of str): List of CSV file paths containing data.
    aggregation_methods (list of str): List of aggregation methods used.
    admet (str): String specifying the ADMET endpoint.
    """
    colors = ['grey', '#8c510a', '#01665e']
    
    for csv_file_path, name_agg, color in zip(csv_file_paths, aggregation_methods, colors):
        df_t = pd.read_csv(csv_file_path)

        dt_all = df_t.copy()
        dt_all['Mean_SR'] *= 100
        dt_all['25th_Percentile_SR'] *= 100
        dt_all['75th_Percentile_SR'] *= 100
        dt_all['25th_Percentile_SR'] = dt_all['Mean_SR'] -  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2
        dt_all['75th_Percentile_SR'] = dt_all['Mean_SR'] +  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2

        dt_all['25th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
        dt_all['75th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
        
        
        # Filter for Chemist Group == 3
        subset = dt_all[dt_all['Chemist Group'] == 3].drop_duplicates("Key")
        keys_with_zero_n = np.insert([ik+1 for ik in range(len(subset))], 0, 0)

        smoothed_data = scipy.ndimage.filters.gaussian_filter1d(subset['Mean_SR'], sigma=1)
        smoothed_data = np.insert(smoothed_data, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data, label=name_agg, linewidth=2, color=color)

        smoothed_data_up = scipy.ndimage.filters.gaussian_filter1d(subset['75th_Percentile_SR'], sigma=1)
        smoothed_data_up = np.insert(smoothed_data_up, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data_up, linewidth=1, color=colors[aggregation_methods.index(name_agg)], linestyle='--')

        smoothed_data_down = scipy.ndimage.filters.gaussian_filter1d(subset['25th_Percentile_SR'], sigma=1)
        smoothed_data_down = np.insert(smoothed_data_down, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data_down, linewidth=1, color=colors[aggregation_methods.index(name_agg)], linestyle='--')
        ax.fill_between(keys_with_zero_n, smoothed_data_up, smoothed_data_down, color=colors[aggregation_methods.index(name_agg)], alpha=0.1)


    ax.set_xlabel('Number of participants', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim((0, 92))
    ax.set_ylim((20, 100))  # Adjust as needed
    ax.set_axisbelow(True)
    plt.tight_layout()



    
def plot_aggregation_expert_group(csv_file_path, output_file_path, name_agg, admet):
    """Plot aggregation of ADMET data for expert groups.

    Args:
    csv_file_path (str): Path to the CSV file containing data.
    output_file_path (str): Path to save the output plot.
    name_agg (str): Aggregation method name.
    admet (str): String specifying the ADMET endpoint.
    """
    df = pd.read_csv(csv_file_path)
    df *= 100  # Scale percentages
    fig, ax = plt.subplots(figsize=(6, 6))

    # Adjust percentiles to center around the mean
    df['25th_Percentile_SR'] = df['Mean_SR'] - abs(df['25th_Percentile_SR'] - df['Mean_SR']) / 2
    df['75th_Percentile_SR'] = df['Mean_SR'] + abs(df['25th_Percentile_SR'] - df['Mean_SR']) / 2
    df[['25th_Percentile_SR', '75th_Percentile_SR']].fillna(df['Mean_SR'], inplace=True)

    chemist_groups = df['Chemist Group'].unique()
    colors = ['#8e0052', '#276319', '#6f9fc7']  # Color for each group

    for i, group in enumerate(chemist_groups):
        subset = df[df['Chemist Group'] == group].drop_duplicates("Key")
        keys_with_zero_n = np.insert([j + 1 for j in range(len(subset))], 0, 0)
        label = {1: "Non-Expert (1-2)", 2: "Expert (3-5)", 3: "All"}.get(group)

        # Smooth data
        for percentile in ['Mean_SR', '75th_Percentile_SR', '25th_Percentile_SR']:
            smoothed_data = scipy.ndimage.filters.gaussian_filter1d(subset[percentile], sigma=1)
            smoothed_data = np.insert(smoothed_data, 0, subset[percentile].iloc[0])
            style = '--' if 'Percentile' in percentile else '-'
            ax.plot(keys_with_zero_n, smoothed_data, label=label if percentile == 'Mean_SR' else None,
                    linewidth=1 if 'Percentile' in percentile else 2, color=colors[i], linestyle=style)

            if 'Percentile' in percentile:
                ax.fill_between(keys_with_zero_n, subset['75th_Percentile_SR'], subset['25th_Percentile_SR'], color=colors[i], alpha=0.1)

    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlabel('Number of Participants', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
    plt.xlim((0, 92))
    plt.ylim((20, 100))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_file_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')

    









def compute_most_frequent(df, weighted=False):
    weights = {1: 1/5, 2: 2/5, 3: 3/5, 4: 4/5, 5: 1}
    # weights = {1: 0/5, 2: 1/5, 3: 2/5, 4: 3/5, 5: 1}

    def weighted_most_frequent(group):
        answers = group['Answer'].unique()
        max_weighted_answer = None
        max_weight = -np.inf

        for answer in answers:
            subset = group[group['Answer'] == answer]
            if weighted:
                weight_sum = (1 * subset['Certitude'].map(weights)).sum()
            else:
                weight_sum = len(subset)
            if weight_sum > max_weight:
                max_weight = weight_sum
                max_weighted_answer = answer
                
        return max_weighted_answer

    group = df.groupby(['Chemist Group', 'Question']).apply(weighted_most_frequent).reset_index(name='Most_Frequent_Answer')
    
    group = group.merge(df[['Question', 'Correct_Answer']].drop_duplicates(), on='Question', how='left')
    group['Most_Frequent_Correct'] = (group['Most_Frequent_Answer'] == group['Correct_Answer']).astype(int)
    
    return group




def compute_most_frequent_combined_weight(df):
    def weighted_combined_most_frequent(group):
        answers = group['Answer'].unique()
        max_weighted_answer = None
        max_weight = -np.inf

        for answer in answers:
            subset = group[group['Answer'] == answer]
            weight_sum = ((subset['Certitude'] + subset['Chemist Level']) / 10).sum()
            if weight_sum > max_weight:
                max_weight = weight_sum
                max_weighted_answer = answer

        return max_weighted_answer

    group = df.groupby(['Chemist Group', 'Question']).apply(weighted_combined_most_frequent).reset_index(name='Most_Frequent_Answer')
    group = group.merge(df[['Question', 'Correct_Answer']].drop_duplicates(), on='Question', how='left')
    group['Most_Frequent_Correct'] = (group['Most_Frequent_Answer'] == group['Correct_Answer']).astype(int)
    return group







def plot_distribution_of_scores(df_A, df_B, output_file_path):
    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
    combined_scores = pd.concat([compute_scores(df_A), compute_scores(df_B)], ignore_index=True)
    combined_scores_all = combined_scores.copy()
    combined_scores_all["Chemist Level"] = 6
    combined_scores = pd.concat([combined_scores_all, combined_scores])
    df_A_all = df_A.copy()
    df_A_all["Chemist Level"] = 6
    df_A = pd.concat([df_A_all, df_A])
    df_B_all = df_B.copy()
    df_B_all["Chemist Level"] = 6
    df_B = pd.concat([df_B_all, df_B])
    df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group)
    df_B["Chemist Group"] = df_B["Chemist Level"].apply(assign_chemist_group)
    combined_scores["Chemist Group"] = combined_scores["Chemist Level"].apply(assign_chemist_group)
    most_frequent_combined = pd.concat([compute_most_frequent(df_A, False), compute_most_frequent(df_B, False)])
    most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
    most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
    most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
    most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
    most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
    most_frequent_combined
    # ... (Include other data processing steps here)

    # Plotting logic
    fig, ax = plt.subplots(figsize=(6, 5))
    chemist_levels = sorted(df_A['Chemist Group'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(chemist_levels)))

    for i, level in enumerate(chemist_levels):
        score_values = combined_scores[combined_scores['Chemist Group'] == level]['Score']
        if score_values.empty:
            continue
        vp = ax.violinplot(score_values, positions=[i], widths=0.9, showextrema=False)
        for pc_idx, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        bp = ax.boxplot(score_values, positions=[i], patch_artist=True, notch=True, widths=0.2, whis=0.5,
                        flierprops={'marker': 'o', 'markersize': 4})
        for box in bp['boxes']:
            box.set(facecolor='darkgrey')
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
            # ... (Include your violin and boxplot logic here)

    # Scatter plot for success rates
    most_frequent_A = df_A.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
    most_frequent_B = df_B.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()

    most_frequent_A['Most_Frequent_Correct'] = np.where(most_frequent_A['Correct_Answer'] == most_frequent_A['Answer'], 1, 0)
    most_frequent_B['Most_Frequent_Correct'] = np.where(most_frequent_B['Correct_Answer'] == most_frequent_B['Answer'], 1, 0)


    most_frequent_combined = pd.concat([most_frequent_A, most_frequent_B])  # Assuming most_frequent_A and most_frequent_B are calculated
    most_frequent_combined = most_frequent_combined.groupby('Chemist Group').agg(
        success_rate=pd.NamedAgg(column='Most_Frequent_Correct', aggfunc='mean')
    ).reset_index()
    most_frequent_combined = most_frequent_combined.rename(columns={"success_rate": "SR"})
    ax.scatter(most_frequent_combined['Chemist Group']-1, most_frequent_combined['SR'], color='white', edgecolor='black', linewidth=1.5, zorder=4, alpha=1, marker="o", s=80)
    # Set axis labels, titles, and grid
    ax.set_xlabel('Chemist Group', fontsize=14)
    chemist_levels = ["Non-Expert", "Expert", "All"]
    ax.set_xticklabels(chemist_levels)
    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('Distribution of Scores per Level', fontsize=16)
    ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
    ax.set_axisbelow(True)
    ax.set_ylim((0, 1))
    plt.tight_layout()
    plt.show()

    fig.savefig(output_file_path, dpi=300, bbox_inches='tight')


    
def plot_distribution_of_scores_session(df_A, output_file_path):
    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    combined_scores = pd.concat([compute_scores(df_A)], ignore_index=True)
    combined_scores_all = combined_scores.copy()
    combined_scores_all["Chemist Level"] = 6
    combined_scores = pd.concat([combined_scores_all, combined_scores])
    df_A_all = df_A.copy()
    df_A_all["Chemist Level"] = 6
    df_A = pd.concat([df_A_all, df_A])
    df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group)
    combined_scores["Chemist Group"] = combined_scores["Chemist Level"].apply(assign_chemist_group)
    most_frequent_combined = pd.concat([compute_most_frequent(df_A, False)])
    most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
    most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
    most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
    most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
    most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
    most_frequent_combined
    # ... (Include other data processing steps here)
    most_frequent_A = df_A.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()

    most_frequent_A['Most_Frequent_Correct'] = np.where(most_frequent_A['Correct_Answer'] == most_frequent_A['Answer'], 1, 0)

    # Plotting logic
    fig, ax = plt.subplots(figsize=(6, 5))
    chemist_levels = sorted(df_A['Chemist Group'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(chemist_levels)))

    for i, level in enumerate(chemist_levels):
        score_values = combined_scores[combined_scores['Chemist Group'] == level]['Score']
        if score_values.empty:
            continue
        vp = ax.violinplot(score_values, positions=[i], widths=0.9, showextrema=False)
        for pc_idx, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        bp = ax.boxplot(score_values, positions=[i], patch_artist=True, notch=True, widths=0.2, whis=0.5,
                        flierprops={'marker': 'o', 'markersize': 4})
        for box in bp['boxes']:
            box.set(facecolor='darkgrey')
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
            # ... (Include your violin and boxplot logic here)

    # Scatter plot for success rates
    most_frequent_combined = pd.concat([most_frequent_A])  # Assuming most_frequent_A and most_frequent_B are calculated
    most_frequent_combined = most_frequent_combined.groupby('Chemist Group').agg(
        success_rate=pd.NamedAgg(column='Most_Frequent_Correct', aggfunc='mean')
    ).reset_index()
    most_frequent_combined = most_frequent_combined.rename(columns={"success_rate": "SR"})
    ax.scatter(most_frequent_combined['Chemist Group']-1, most_frequent_combined['SR'], color='white', edgecolor='black', linewidth=1.5, zorder=4, alpha=1, marker="o", s=80)
    # Set axis labels, titles, and grid
    ax.set_xlabel('Chemist Group', fontsize=14)
    chemist_levels = ["Non-Expert", "Expert", "All"]
    ax.set_xticklabels(chemist_levels)
    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('Distribution of Scores per Level', fontsize=16)
    ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
    ax.set_axisbelow(True)
    ax.set_ylim((0, 1))
    plt.tight_layout()
    plt.show()

    fig.savefig(output_file_path, dpi=300, bbox_inches='tight')


    

def assign_chemist_group_S1A(level):
    if level < 3:
        return 1  # non-expert
    elif level >=3 and level <= 5:
        return 2  # expert
    else:
        return 3  # all

    
    
def assign_chemist_group_spe_S1B(level):
    if level < 3:
        return 1  # non-expert
    elif level ==3:
        return 2  # expert

    elif level >4 and level <= 5:
        return 3  # expert
    else:
        return 4  # all
    
    


def plot_success_rate_by_endpoint(csv_file_path_A, csv_file_path_B, output_file_path):
	# Read the CSV files
	df_A = pd.read_csv(csv_file_path_A, sep=',')
	df_B = pd.read_csv(csv_file_path_B, sep=',')
	df_B = remove_consistent_chemists(df_B)
	df_A = remove_consistent_chemists(df_A)
	df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
	df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
	df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group)
	df_B["Chemist Group"] = df_B["Chemist Level"].apply(assign_chemist_group)
	# Combine both dataframes
	combined_df = pd.concat([df_A, df_B])
	combined_df = combined_df[combined_df['Certitude'].isna()!=True]
	# Calculate success rate by endpoint for users
	combined_df_expert = combined_df[combined_df["Chemist Group"] == 2]
	user_success_rate_by_endpoint = combined_df.groupby(['Chemist', 'Endpoint'])['Result'].mean().reset_index()
	user_success_rate_by_endpoint_expert = combined_df_expert.groupby(['Chemist', 'Endpoint'])['Result'].mean().reset_index()
	user_success_rate_by_endpoint = user_success_rate_by_endpoint[user_success_rate_by_endpoint["Result"]!=0]
	user_success_rate_by_endpoint_expert = user_success_rate_by_endpoint_expert[user_success_rate_by_endpoint_expert["Result"]!=0]
	user_success_rate_by_endpoint = user_success_rate_by_endpoint.drop_duplicates()
	user_success_rate_by_endpoint_expert = user_success_rate_by_endpoint_expert.drop_duplicates()
	most_frequent_A_endpoint = df_A.groupby(['Endpoint', 
		'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_A_endpoint['Most_Frequent_Correct'] = np.where(most_frequent_A_endpoint['Correct_Answer'] == most_frequent_A_endpoint['Answer'], 1, 0)
	most_frequent_B_endpoint = df_B.groupby(['Endpoint', 
		'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_B_endpoint['Most_Frequent_Correct'] = np.where(most_frequent_B_endpoint['Correct_Answer'] == most_frequent_B_endpoint['Answer'], 1, 0)
	most_frequent_combined_endpoint = pd.concat([most_frequent_A_endpoint, most_frequent_B_endpoint])
	most_frequent_success_rate_by_endpoint = most_frequent_combined_endpoint.groupby('Endpoint')['Most_Frequent_Correct'].mean().reset_index()
	ordered_endpoints = ["LogP", "Permeability", "Solubility", "LogD", "hERG"][::-1]
	colors = plt.cm.viridis(np.linspace(0, 1, len(ordered_endpoints)))
	###
	most_frequent_combined_weighted = pd.concat([compute_most_frequent_Endpoint(df_A, True), 
		compute_most_frequent_Endpoint(df_B, True)]).groupby('Endpoint')['Most_Frequent_Correct'].mean().reset_index(name='Most_Frequent_Correct')
	most_frequent_combined_weighted_both = pd.concat([compute_most_frequent_combined_weight_Endpoint(df_A), 
		compute_most_frequent_combined_weight_Endpoint(df_B)]).groupby('Endpoint')['Most_Frequent_Correct'].mean().reset_index(name='Most_Frequent_Correct')
	most_frequent_combined_weighted["Most_Frequent_Correct"]*=100
	most_frequent_combined_weighted_both["Most_Frequent_Correct"]*=100
	# Plotting
	fig1, ax1 = plt.subplots(figsize=(5.5, 5))
	ordered_endpoints = ["LogP", "Permeability", "Solubility", "LogD", "hERG"][::-1]
	colors = plt.cm.viridis(np.linspace(0, 1, len(ordered_endpoints)))
	colors = plt.get_cmap('Dark2').colors  # Using colors from the Set3 colormap
	ordered_endpoints=["LogP", "Permeability", "Solubility", "LogD", "hERG"][::-1]
	for idx, endpoint in enumerate(ordered_endpoints):
		data = user_success_rate_by_endpoint[user_success_rate_by_endpoint['Endpoint'] == endpoint]['Result']
		data_expert = user_success_rate_by_endpoint_expert[user_success_rate_by_endpoint_expert['Endpoint'] == endpoint]['Result']
		data*=100
		data_expert*=100
		vp = ax1.violinplot(data, positions=[idx], widths=0.5, showextrema=False)
		for pc in vp['bodies']:
			pc.set_facecolor(colors[idx])
			pc.set_alpha(1)
		bp = ax1.boxplot(data, positions=[idx], patch_artist=True, notch=True, widths=0.2, whis=0.5,
						 flierprops={'marker': 'o', 'markersize': 4})
		for box in bp['boxes']:
			box.set(facecolor='darkgrey')
		for median in bp['medians']:
			median.set(color='black', linewidth=2)
		most_frequent_success_rate = most_frequent_success_rate_by_endpoint[most_frequent_success_rate_by_endpoint['Endpoint'] == endpoint]['Most_Frequent_Correct'].values[0]
		most_frequent_success_rate_w = most_frequent_combined_weighted[most_frequent_combined_weighted['Endpoint'] == endpoint]['Most_Frequent_Correct'].values[0]
		most_frequent_most_frequent_combined_weighted_both = most_frequent_combined_weighted_both[most_frequent_combined_weighted_both['Endpoint'] == endpoint]['Most_Frequent_Correct'].values[0]
		most_frequent_success_rate = most_frequent_success_rate*100
		ax1.scatter(idx, most_frequent_success_rate, color='white', edgecolor='black', linewidth=1.5, zorder=4, alpha=1, marker="o", s=80, 
							   label='Collective Intelligence' if idx == 0 else "")
	ax1.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax1.set_axisbelow(True)
	ax1.set_xticks(range(len(ordered_endpoints)))
	ax1.set_xticklabels(ordered_endpoints, fontsize=10)
	ax1.set_xlabel('ADMET Endpoint', fontsize=14)
	ax1.set_ylabel('Success Rate (%)', fontsize=14)
	ax1.set_ylim((0, 100))
	ax1.legend(loc='upper left', fontsize=12, framealpha = 1)
	plt.tight_layout()
	fig1.savefig(output_file_path, dpi=300, bbox_inches='tight')
	fig1.savefig(output_file_path.replace(".png",".svg"), dpi=300, bbox_inches='tight')
	plt.show()

	

def plot_certitude_success_distribution_color(df, output_file_path):
	# Process the DataFrame for plotting
	df["Certitude"] = df["Certitude"].astype(str)
	success_rate = df.groupby(['Chemist Level', 'Certitude'])['Result'].mean().reset_index(name='Success Rate')
	bubble_data = df.groupby(['Chemist Level', 'Certitude']).size().reset_index(name='Counts')
	bubble_data = bubble_data.merge(success_rate, on=['Chemist Level', 'Certitude'])
	bubble_data["Success Rate"] = (bubble_data["Success Rate"] * 100).astype(int)

	# Normalize success rate for color mapping between 0 and 100
	norm = Normalize(vmin=20, vmax=80)

	# Create the plot
	fig, ax = plt.subplots(figsize=(6, 5))
	scatter = ax.scatter(bubble_data['Chemist Level'], bubble_data['Certitude'], 
						 s=bubble_data['Counts'] * 2, alpha=1, c=bubble_data['Success Rate'], 
						 edgecolors='k', linewidth=1, cmap='RdYlGn', norm=norm)

	# Set labels and title
	ax.set_xlabel('Expertise', fontsize=14)
	ax.set_ylabel('Confidence', fontsize=14)

	# Create color bar
	cbar = plt.colorbar(scatter, ax=ax)
	cbar.set_label('Success Rate (%)', fontsize=14)
	cbar.set_ticks(np.arange(20, 81, 10))  # Setting ticks at regular intervals

	# Show and save the plot
	fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
	fig.savefig(output_file_path.replace(".png",".svg"), dpi=300, bbox_inches='tight')

	plt.show()



def plot_aggregation_S7(ax, csv_file_path, name_agg, admet, i, j):
	df_t = pd.read_csv(csv_file_path)
	dt_all = df_t.copy()
	dt_all['Mean_SR']*=100
	dt_all['25th_Percentile_SR']*=100
	dt_all['75th_Percentile_SR']*=100
	dt_all['25th_Percentile_SR'] = dt_all['Mean_SR'] -  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2
	dt_all['75th_Percentile_SR'] = dt_all['Mean_SR'] +  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2
	dt_all['25th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
	dt_all['75th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
	chemist_groups = list(set(list(dt_all["Chemist Group"])))
	for unique_group in chemist_groups:
		group_df = dt_all[dt_all["Chemist Group"] == unique_group]
		group_df = group_df[group_df["Key"]==20]
	colors = plt.cm.brg(np.linspace(0., 1, len(chemist_groups)))
	# plt.style.use('ggplot')
	plt.style.use('default')
	# Darken the colors
	colors =['#94568c', '#65ab7c', '#2976bb']
	colors =['#8e0052', '#276319', '#6f9fc7']
	k = 0
	for group in chemist_groups:
		subset = dt_all[dt_all['Chemist Group'] == group]
		subset = subset.drop_duplicates("Key")
		keys_with_zero_n = np.insert([ik+1 for ik in range(len(subset))], 0, 0)
		label = {1: "Non Expert (1-2)", 2: "Expert (3-5)", 3: "All"}.get(group)
		smoothed_data = scipy.ndimage.filters.gaussian_filter1d(subset['Mean_SR'], sigma=1)
		smoothed_data = np.insert(smoothed_data, 0, subset["Mean_SR"].tolist()[0])
		ax.plot(keys_with_zero_n, smoothed_data, label=label, linewidth=2, color=colors[k])
		smoothed_data_up = scipy.ndimage.filters.gaussian_filter1d(subset['75th_Percentile_SR'], sigma=1)
		smoothed_data_up = np.insert(smoothed_data_up, 0, subset["Mean_SR"].tolist()[0])
		ax.plot(keys_with_zero_n, smoothed_data_up, linewidth=1, color=colors[k], linestyle='--')
		smoothed_data_down = scipy.ndimage.filters.gaussian_filter1d(subset['25th_Percentile_SR'], sigma=1)
		smoothed_data_down = np.insert(smoothed_data_down, 0, subset["Mean_SR"].tolist()[0])
		ax.plot(keys_with_zero_n, smoothed_data_down, linewidth=1, color=colors[k], linestyle='--')
		ax.fill_between(keys_with_zero_n, smoothed_data_up, smoothed_data_down, color=colors[k], alpha=0.1)
		k += 1

	# Set x-axis label only for the last row
	if i == 5:  # assuming 6 rows, adjust as needed
		ax.set_xlabel('Number of participants', fontsize=14)
	
	
	d_a = {"most_frequent":"Most Frequent",
	"log_odds":"Log Odds",
	"weighted_by_certitude":"Weighted By Certitude",
	"weighted_by_certitude_and_expertise":"Weighted By Certitude & Expertise",
	"weighted_by_expertise":"Weighted By Expertise",
	"fuzzy_logic_aggregation":"Fuzzy Logic Aggregation"}

	# Set y-axis label only for the first column
	ax.set_title(d_a[name_agg], fontsize=14)

	# Set x-axis label only for the last row
	if j == 0:  # assuming 6 rows, adjust as needed
		ax.set_ylabel("Success Rate (%)", fontsize=14)
	
	# Set y-axis label only for the first column
	# if j == 0:
	#     ax.set_ylabel('Success Rate (%)', fontsize=14)

	ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax.set_xlim((0, 92))
	if 'LogP' in admet:
		ax.set_ylim(30, 100)
	else:
		ax.set_ylim(20, 70)
	
	ax.set_axisbelow(True)
	# plt.tight_layout()
	# plt.show()
	# fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
	return ax



def plot_distribution_of_scores_fig1c(df_A, df_B, output_file_path):
	df_B = remove_consistent_chemists(df_B)
	df_A = remove_consistent_chemists(df_A)
	df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
	df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
	combined_scores = pd.concat([compute_scores(df_A), compute_scores(df_B)], ignore_index=True)
	combined_scores_all = combined_scores.copy()
	combined_scores_all["Chemist Level"] = 6
	combined_scores = pd.concat([combined_scores_all, combined_scores])
	df_A_all = df_A.copy()
	df_A_all["Chemist Level"] = 6
	df_A = pd.concat([df_A_all, df_A])
	df_B_all = df_B.copy()
	df_B_all["Chemist Level"] = 6
	df_B = pd.concat([df_B_all, df_B])
	df_A["Chemist Group"] = df_A["Chemist Level"].tolist()
	df_B["Chemist Group"] = df_B["Chemist Level"].tolist()
	combined_scores["Chemist Group"] = combined_scores["Chemist Level"].tolist()
	most_frequent_combined = pd.concat([compute_most_frequent(df_A, False), compute_most_frequent(df_B, False)])
	most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
	most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
	most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
	most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
	most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
	most_frequent_combined

	# Plotting logic
	fig, ax = plt.subplots(figsize=(5.5, 5))
	chemist_levels = sorted(df_A['Chemist Group'].unique())

	# Custom colors: PiYG for Non-Expert and Expert, specific color for All
	colors = plt.get_cmap('PiYG')(np.linspace(0., 1, len(chemist_levels) - 1)).tolist()
	colors.append('#2976bb')  # Adding the specific color for the 'All' group
	add_label = True
	colors[2] = [0.85, 0.85, 0.85, 1]  # Dark grey color with full opacity

	# Scatter plot for success rates
	most_frequent_A = df_A.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_B = df_B.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_A['Most_Frequent_Correct'] = np.where(most_frequent_A['Correct_Answer'] == most_frequent_A['Answer'], 1, 0)
	most_frequent_B['Most_Frequent_Correct'] = np.where(most_frequent_B['Correct_Answer'] == most_frequent_B['Answer'], 1, 0)
	most_frequent_combined = pd.concat([most_frequent_A, most_frequent_B])  # Assuming most_frequent_A and most_frequent_B are calculated
	most_frequent_combined = most_frequent_combined.groupby('Chemist Group').agg(
		success_rate=pd.NamedAgg(column='Most_Frequent_Correct', aggfunc='mean')
	).reset_index()
	most_frequent_combined = most_frequent_combined.rename(columns={"success_rate": "SR"})

	for i, level in enumerate(chemist_levels):
		score_values = combined_scores[combined_scores['Chemist Group'] == level]['Score']
		score_values *=100
		if score_values.empty:
			continue
		vp = ax.violinplot(score_values, positions=[i], widths=0.9, showextrema=False)
		for pc_idx, pc in enumerate(vp['bodies']):
			pc.set_facecolor(colors[i])
			pc.set_alpha(1)
		bp = ax.boxplot(score_values, positions=[i], patch_artist=True, notch=True, widths=0.2, whis=0.5,
						flierprops={'marker': 'o', 'markersize': 4})
		for box in bp['boxes']:
			box.set(facecolor='darkgrey')
		for median in bp['medians']:
			median.set(color='black', linewidth=2)

		# Scatter plot for success rates
		if add_label:  # Adding label only for the first scatter plot
			scatter_label = 'Collective Intelligence'
			add_label = False  # Reset the flag so the label is not added again
		else:
			scatter_label = None

		most_frequent_combined = pd.concat([most_frequent_A, most_frequent_B])  
		most_frequent_combined = most_frequent_combined.groupby('Chemist Group').agg(
			success_rate=pd.NamedAgg(column='Most_Frequent_Correct', aggfunc='mean')
		).reset_index()
		most_frequent_combined = most_frequent_combined.rename(columns={"success_rate": "SR"})

		ax.scatter(most_frequent_combined['Chemist Group']-1, most_frequent_combined['SR']*100, 
				   color='white', edgecolor='black', linewidth=1.5, zorder=4, alpha=1, 
				   marker="o", s=80, label=scatter_label)
	ax.scatter(most_frequent_combined['Chemist Group']-1, most_frequent_combined['SR']*100, color='white', 
							   label='Collective Intelligence' if i == 0 else "", 
			   edgecolor='black', linewidth=1.5, zorder=4, alpha=1, marker="o", s=80)
	# Set axis labels, titles, and grid
	ax.set_xlabel('Expertise', fontsize=14)
	chemist_levels = ["Non-Expert (1-2)", "Expert (3-5)", "All"]
	ax.set_xticklabels(["1","2","3","4","5","All"])
	ax.set_ylabel('Success Rate (%)', fontsize=14)
	ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax.set_axisbelow(True)
	ax.set_ylim((0, 100))
	ax.legend(loc='upper left', fontsize=12, framealpha = 1)

	plt.tight_layout()

	fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
	fig.savefig(output_file_path.replace(".png",".svg"), dpi=300, bbox_inches='tight')
	plt.show()

	
	
def plot_distribution_of_scores_S1B(df_A, df_B, output_file_path):
	df_B = remove_consistent_chemists(df_B)
	df_A = remove_consistent_chemists(df_A)
	df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
	df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)
	combined_scores = pd.concat([compute_scores(df_A), compute_scores(df_B)], ignore_index=True)
	combined_scores_all = combined_scores.copy()
	combined_scores_all["Chemist Level"] = 6
	combined_scores = pd.concat([combined_scores_all, combined_scores])
	df_A_all = df_A.copy()
	df_A_all["Chemist Level"] = 6
	df_A = pd.concat([df_A_all, df_A])
	df_B_all = df_B.copy()
	df_B_all["Chemist Level"] = 6
	df_B = pd.concat([df_B_all, df_B])
	df_A["Chemist Group"] = df_A["Chemist Level"].apply(assign_chemist_group_spe_S1B)
	df_B["Chemist Group"] = df_B["Chemist Level"].apply(assign_chemist_group_spe_S1B)
	combined_scores["Chemist Group"] = combined_scores["Chemist Level"].apply(assign_chemist_group_spe_S1B)
	most_frequent_combined = pd.concat([compute_most_frequent(df_A, False), compute_most_frequent(df_B, False)])
	most_frequent_combined = most_frequent_combined[most_frequent_combined["Chemist Group"]==3]
	most_frequent_combined = most_frequent_combined[["Question","Most_Frequent_Answer"]]
	most_frequent_combined.columns = ["Slide_ID","Most_Frequent_Answer"]
	most_frequent_combined["Slide_ID"] = [i.split("Q")[-1] for i in most_frequent_combined["Slide_ID"].tolist()]
	most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv", index = False)
	most_frequent_combined

	# Plotting logic
	fig, ax = plt.subplots(figsize=(5.5, 5))
	chemist_levels = sorted(df_A['Chemist Group'].unique())

	# Custom colors: PiYG for Non-Expert and Expert, specific color for All
	colors = plt.get_cmap('PiYG')(np.linspace(0., 1, len(chemist_levels) - 1)).tolist()
	colors.append('#2976bb')  # Adding the specific color for the 'All' group
	colors[1] = [0.85, 0.85, 0.85, 1]  # Dark grey color with full opacity
	for i, level in enumerate(chemist_levels):
		score_values = combined_scores[combined_scores['Chemist Group'] == level]['Score']
		score_values *=100
		if score_values.empty:
			continue
		vp = ax.violinplot(score_values, positions=[i], widths=0.9, showextrema=False)
		for pc_idx, pc in enumerate(vp['bodies']):
			pc.set_facecolor(colors[i])
			pc.set_alpha(1)
		bp = ax.boxplot(score_values, positions=[i], patch_artist=True, notch=True, widths=0.2, whis=0.5,
						flierprops={'marker': 'o', 'markersize': 4})
		for box in bp['boxes']:
			box.set(facecolor='darkgrey')
		for median in bp['medians']:
			median.set(color='black', linewidth=2)
	# Scatter plot for success rates
	most_frequent_A = df_A.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_B = df_B.groupby(['Chemist Group', 'Question']).agg(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_A['Most_Frequent_Correct'] = np.where(most_frequent_A['Correct_Answer'] == most_frequent_A['Answer'], 1, 0)
	most_frequent_B['Most_Frequent_Correct'] = np.where(most_frequent_B['Correct_Answer'] == most_frequent_B['Answer'], 1, 0)
	most_frequent_combined = pd.concat([most_frequent_A, most_frequent_B])  # Assuming most_frequent_A and most_frequent_B are calculated
	most_frequent_combined.to_csv("./data/CollectiveIntelligence/CI_response_Human.csv", index = False)
	most_frequent_combined = most_frequent_combined.groupby('Chemist Group').agg(
		success_rate=pd.NamedAgg(column='Most_Frequent_Correct', aggfunc='mean')
	).reset_index()
	most_frequent_combined = most_frequent_combined.rename(columns={"success_rate": "SR"})
	ax.scatter(most_frequent_combined['Chemist Group']-1, most_frequent_combined['SR']*100, color='white', edgecolor='black', linewidth=1.5, zorder=4, alpha=1, marker="o", s=80)
	# Set axis labels, titles, and grid
	ax.set_xlabel('Expertise Group', fontsize=14)
	chemist_levels = ["Non-Expert (1-2)", "Average (3)", "Expert (4-5)", "All"]
	ax.set_xticklabels(chemist_levels)
	ax.set_ylabel('Success Rate (%)', fontsize=14)
	ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax.set_axisbelow(True)
	ax.set_ylim((0, 100))
	plt.tight_layout()

	fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
	fig.savefig(output_file_path.replace(".png",".svg"), dpi=300, bbox_inches='tight')
	plt.show()

	
	
def darken_color(color, factor=0.6):
	'''
	Darkens the given color by a specified factor.

	Parameters:
	- color: A list of RGB values (e.g., [255, 255, 255]).
	- factor: A float value by which to darken the color (default is 0.6).

	Returns:
	- A list of RGB values representing the darkened color.
	'''
	return [max(0, int(c * factor)) for c in color]


def _waffle_plot(df_a, df_b):
	import matplotlib.patches as mpatches

	df_combined = pd.concat([df_a, df_b])
	# Extracting the float number from the 'Question' column
	df_combined['Question_Number'] = df_combined['Question'].apply(lambda x: float(re.search(r'Q(\d+\.\d+|\d+)', x).group(1)))
	# Removing duplicates based on 'Question'
	df_combined = df_combined.drop_duplicates(subset=['Question'])
	# Sorting the dataframe by 'Question_Number'
	df_sorted = df_combined.sort_values(by='Question_Number')
	# Create a color map for each unique 'Endpoint'
	unique_endpoints = df_sorted['Endpoint'].unique()
	colors = plt.get_cmap('Set3').colors  # Using colors from the Set3 colormap
	color_map = dict(zip(unique_endpoints, colors))
	# Endpoint to color mapping
	endpoint_to_color = {
		'LogP': "#66a61e",
		'Permeability': "#e7298a",
		'Solubility':"#7570b3",
		'LogD': "#d95f02",
		'hERG': "#1b9e77"
	}
	# Update the color map for each unique 'Endpoint'
	color_map = {endpoint: endpoint_to_color.get(endpoint, (1, 1, 1, 1)) for endpoint in df_sorted['Endpoint'].unique()}
	# Update the list of colors for each row in df_sorted
	sorted_colors = df_sorted['Endpoint'].map(color_map).tolist()
	# The values for each square in the waffle chart (1 square per question)
	values = [1] * len(df_sorted)
	# Draw the Waffle Chart
	fig = plt.figure(
		FigureClass=Waffle,
		plots={
			111: {
				'values': values,
				'colors': sorted_colors,
				'title': {'label': '', 'loc': 'center', 'fontsize': 18}
			},
		},
		rows=19,  # Adjust the number of rows as needed
		figsize=(9, 9)
	)
	# Create custom legend
	patches = [mpatches.Patch(color=color_map[endpoint], label=endpoint) for endpoint in unique_endpoints]
	plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title="ADMET Endpoint", title_fontsize = 12)
	fig.savefig("./figure/Figure_1a.svg", format='svg')
	fig.savefig("./figure/Figure_1a.png", format='png')
	plt.show()

def _distribution_groups():
	'''
	Generates a donut chart displaying the distribution of groups.
	'''
	labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5']
	sizes = [31, 24, 19, 9, 9]
	total = sum(sizes)
	colors = plt.get_cmap('PiYG')(np.linspace(0., 1, len(sizes)))
	colors[2] = [0.85, 0.85, 0.9, 1]  # Dark grey color with full opacity

	fig, ax = plt.subplots(figsize=(5, 5))
	wedges, texts = ax.pie(sizes, labels=labels, labeldistance=1.25, colors=colors, startangle=90,
						   wedgeprops={'edgecolor': 'black', 'linewidth': 1})

	for i, (label, pie_wedge) in enumerate(zip(texts, wedges)):
		x, y = label.get_position()
		percent = f"{100 * sizes[i] / total:.1f}%"
		ax.text(x, y - 0.1, percent, ha='center', va='top', fontsize='11')

	for text in texts:
		text.set_horizontalalignment('center')
		text.set_fontsize(11)

	ax.add_artist(plt.Circle((0, 0), 0.6, color='black', edgecolor='black', linewidth=1))
	ax.add_artist(plt.Circle((0, 0), 0.59, color='white', edgecolor='black', linewidth=1))

	plt.savefig("./figure/Figure_1b.svg", format='svg')
	plt.savefig("./figure/Figure_1b.png", format='png')
	plt.show()



def _distribution_ADMET():
	'''
	Generates a donut chart displaying the distribution of ADMET-related data.
	'''
	labels = ["LogP", "Permeability", "Solubility", "LogD", "hERG"][::-1]
	sizes = [12, 9, 26, 12, 15]
	total = sum(sizes)
	colors = plt.get_cmap('Dark2').colors
	fig, ax = plt.subplots(figsize=(5, 5))
	wedges, texts = ax.pie(sizes, labels=labels, labeldistance=1.25, colors=colors, startangle=221,
						   wedgeprops={'edgecolor': 'black', 'linewidth': 1})
	for i, (label, pie_wedge) in enumerate(zip(texts, wedges)):
		x, y = label.get_position()
		percent = f"{100 * sizes[i] / total:.1f}%"
		ax.text(x, y - 0.1, percent, ha='center', va='top', fontsize='11')

	for text in texts:
		text.set_horizontalalignment('center')
		text.set_fontsize(11)
	ax.add_artist(plt.Circle((0, 0), 0.6, color='black', edgecolor='black', linewidth=1))
	ax.add_artist(plt.Circle((0, 0), 0.59, color='white', edgecolor='black', linewidth=1))
	plt.savefig('./figure/Figure_1e.png', format='png')
	plt.savefig('./figure/Figure_1e.svg', format='svg')
	plt.show()

	
def _bubble_ADMET(combined_df):
	# Process the DataFrame for plotting
	combined_df["Certitude"] = [str(i) for i in combined_df["Certitude"].tolist()]
	success_rate = combined_df.groupby(['Endpoint', 'Certitude'])['Result'].mean().reset_index(name='Success Rate')
	bubble_data = combined_df.groupby(['Endpoint', 'Certitude']).size().reset_index(name='Counts')
	bubble_data = bubble_data.merge(success_rate, on=['Endpoint', 'Certitude'])
	bubble_data["Success Rate"] = [int(float(i)*100) for i in bubble_data["Success Rate"].tolist()]
	ordered_endpoints = ["LogP", "Permeability", "Solubility", "LogD", "hERG"][::-1]
	bubble_data['Endpoint'] = pd.Categorical(bubble_data['Endpoint'], categories=ordered_endpoints, ordered=True)
	bubble_data = bubble_data.sort_values(by=['Endpoint', 'Certitude'])
	# Normalize success rate for color mapping between 0 and 100
	norm = Normalize(vmin=20, vmax=80)

	# Create the plot
	fig1, ax1 = plt.subplots(figsize=(6, 5))
	scatter = ax1.scatter(bubble_data['Endpoint'], bubble_data['Certitude'], 
						  s=bubble_data['Counts']*2, alpha=1, c=bubble_data['Success Rate'], 
						  edgecolors='k', linewidth=1, cmap='RdYlGn', norm=norm)
	ax1.set_xlabel('ADMET Endpoint', fontsize=14)
	ax1.set_ylabel('Confidence', fontsize=14)

	# Create color bar
	cbar = plt.colorbar(scatter, ax=ax1)
	cbar.set_label('Success Rate (%)', fontsize=14)
	cbar.set_ticks(np.arange(20, 81, 10))  # Setting ticks at regular intervals
	fig1.savefig("./figure/Figure_1g.png", dpi=300, bbox_inches='tight')

	fig1.savefig("./figure/Figure_1g.svg", dpi=300, bbox_inches='tight')
	plt.show()

def _UMAP_SR(result_df_SLIDE_to_SR, endpoint_data):
	sr_column='SR'
	endpoint_column='Endpoint'
	answer_correct_column='Answer_Correct'
	endpoint_data = result_df_SLIDE_to_SR.copy()

	# Group by total_smiles and calculate the mean SR
	mean_sr_per_smiles_to_show = endpoint_data.groupby("total_smiles")["SR"].mean()
	mean_sr_per_smiles_to_show = pd.DataFrame(mean_sr_per_smiles_to_show.reset_index())
	result_df_SLIDE_to_SR_k = endpoint_data.merge(mean_sr_per_smiles_to_show, on = 'total_smiles')
	result_df_SLIDE_to_SR_k = result_df_SLIDE_to_SR_k.drop_duplicates("total_smiles")
	endpoint_data = result_df_SLIDE_to_SR_k.rename(columns = {"SR_x":"SR"})
	norm = Normalize(vmin=20, vmax=80)

	# Extracting the ECFP features and SR values
	success_rate = endpoint_data[sr_column]
	# Plotting
	success_rate*=100
	norm = Normalize(vmin=20, vmax=80)
	fig, ax = plt.subplots(figsize=(6.2, 5))
	# Creating a dummy scatter plot for the colorbar
	scatter = ax.scatter(endpoint_data["TSNE_PC1"].tolist(), endpoint_data["TSNE_PC2"].tolist(), c=success_rate, s=30,
						 cmap='RdYlGn', alpha=1, edgecolors='k', linewidth=1, norm=norm)
	ax.set_xlabel('Principal Component 1', fontsize=14)
	ax.set_ylabel('Principal Component 2', fontsize=14)

	ax.spines['right'].set_visible(True)
	ax.spines['top'].set_visible(True)
	cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
	cbar.set_ticks(np.arange(20, 81, 10))  # Setting ticks at regular intervals

	cbar.set_label('Success Rate (%)', fontsize=14)
	plt.tight_layout()

	fig.savefig("./figure/Figure_3a.png", format='svg')
	fig.savefig("./figure/Figure_3a.svg", format='png')

	plt.show()



def _UMAP_ADMET(result_df_SLIDE_to_SR, endpoint_data):
	sr_column='SR'
	endpoint_column='Endpoint'
	answer_correct_column='Answer_Correct'
	endpoint_data = result_df_SLIDE_to_SR.copy()

	# Group by total_smiles and calculate the mean SR
	mean_sr_per_smiles = endpoint_data.groupby("total_smiles")[sr_column].mean()
	mean_sr_per_smiles = pd.DataFrame(mean_sr_per_smiles.reset_index())
	result_df_SLIDE_to_SR_k = result_df_SLIDE_to_SR.merge(mean_sr_per_smiles, on='total_smiles')
	result_df_SLIDE_to_SR_k = result_df_SLIDE_to_SR_k.drop_duplicates("total_smiles")
	endpoint_data = result_df_SLIDE_to_SR_k.rename(columns={"SR_x": "SR"})

	# Create a colormap for the endpoints
	unique_endpoints = endpoint_data[endpoint_column].unique()
	unique_endpoints = ["LogP", "Permeability", "Solubility", "LogD", "hERG"]
	colors = plt.cm.get_cmap('Set3', len(unique_endpoints))
	# Endpoint to color mapping
	endpoint_to_color = {
		'LogP': "#66a61e",
		'Permeability': "#e7298a",
		'Solubility':"#7570b3",
		'LogD': "#d95f02",
		'hERG': "#1b9e77"
	}
	# Plotting
	fig, ax = plt.subplots(figsize=(5.5, 5))

	# Create a scatter plot for each endpoint
	for endpoint in unique_endpoints:
		ep_data = endpoint_data[endpoint_data[endpoint_column] == endpoint]
		ax.scatter(ep_data["TSNE_PC1"], ep_data["TSNE_PC2"], color=endpoint_to_color[endpoint], 
				   label=endpoint, s=30, alpha=1, edgecolors='k', linewidth=1)

	# Set titles and labels
	ax.set_xlabel('Principal Component 1', fontsize=14)
	ax.set_ylabel('Principal Component 2', fontsize=14)

	ax.spines['right'].set_visible(True)
	ax.spines['top'].set_visible(True)

	# Create a legend
	ax.legend( fontsize=10, loc='best')
	plt.tight_layout()

	fig.savefig("./figure/Figure_3b.png", format='svg')
	fig.savefig("./figure/Figure_3b.svg", format='png')
	plt.show()


	
def rgb_to_hex(rgb):
	'''
	Convert an RGB tuple to a hexadecimal string.

	Args:
	rgb (tuple): A tuple of three float values (r, g, b) ranging from 0 to 1.

	Returns:
	str: Hexadecimal color code.
	'''
	return f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'



def plot_distribution(df_endpoint, npk):
	'''
	Plot distribution of endpoint data and save as image files.

	Args:
	df_endpoint (DataFrame): Data containing the response median.
	npk (str): Name of the endpoint/key to be used in the title and file names.
	'''
	fig, ax1 = plt.subplots(figsize=(5, 4))
	values = df_endpoint["Resp_MEDIAN"].tolist()
	hist_data, bins, patches = ax1.hist(values, bins=8, alpha=1, edgecolor='black', linewidth=1.2)
	bin_centers = 0.5 * (bins[:-1] + bins[1:])
	colors = plt.cm.gray(bin_centers / max(bin_centers))

	for patch, color in zip(patches, colors):
		patch.set_facecolor("darkgrey")

	ax1.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax1.set_axisbelow(True)
	ax1.set_title(npk, fontsize=14)
	ax1.set_xlabel('Experimental Measurement', fontsize=14)
	ax1.set_ylabel('Number of Entries', fontsize=14)
	plt.tight_layout()
	fig.savefig(f"./figure/Figure_S14-{npk}.svg", format='svg')
	fig.savefig(f"./figure/Figure_S14-{npk}.png", format='png')
	plt.show()
	


def plot_hexbin(dataframe, save_id, ax, index, total_cols, gridsize=70, cmap='viridis'):
	'''
	Plot hexbin for experimental vs prediction data and add performance metrics.

	Args:
	dataframe (DataFrame): Data containing 'Experimental' and 'Prediction' columns.
	save_id (str): Identifier for saving and labeling plots.
	ax (AxesSubplot): Matplotlib subplot object to plot on.
	index (int): Index of the subplot in a grid.
	total_cols (int): Total number of columns in subplot grid.
	gridsize (int): Size of the grid for hexbin plot.
	cmap (str): Color map for the hexbin plot.

	Returns:
	AxesSubplot: The matplotlib subplot object.
	'''

	ax.set_facecolor('white')
	ax.grid(axis='y', linestyle='-', color='gray', alpha=0.5)
	ax.grid(axis='x', linestyle='-', color='gray', alpha=0.5)
	ax.set_axisbelow(True)

	# Extract 'Experimental' and 'Prediction' values
	all_experimentals = dataframe['Experimental']
	all_predictions = dataframe['Prediction']

	min_val = min(min(all_experimentals), min(all_predictions))
	max_val = max(max(all_experimentals), max(all_predictions))

	# Calculate metrics
	mae = mean_absolute_error(all_experimentals, all_predictions)
	rmse = np.sqrt(mean_squared_error(all_experimentals, all_predictions))
	r2 = r2_score(all_experimentals, all_predictions)
	print("mae",mae)
	print("rmse",rmse)
	print("r2",r2)

	ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', alpha=0.5, linewidth=2)
	hb = ax.hexbin(all_experimentals, all_predictions, gridsize=gridsize, cmap=cmap, bins='log')
	cb = plt.colorbar(hb, ax=ax)
	if index % total_cols != 0:  # Check if it's not in the first column
		cb.set_label(r'$\log_{10}(\mathrm{Number\;of\;compounds})$', fontsize=14)
	if index == 4:   # Check if it's not in the first column
		cb.set_label(r'$\log_{10}(\mathrm{Number\;of\;compounds})$', fontsize=14)

	ax.set_xlim(min_val, max_val)
	ax.set_ylim(min_val, max_val)
	
	ax.set_xlabel('Experimental', fontsize = 14)
	ax.set_ylabel('Prediction', fontsize = 14)
	if "Papp" in save_id:
		ax.set_title("Permeability", fontsize = 14)
	else:
		if  "Sapp" in save_id:
			ax.set_title("Solubility", fontsize = 14)
		else:
			ax.set_title(save_id, fontsize = 14)

	plt.tight_layout()
	return ax



def _generatre_benchmark(df_list):
	# Concatenate all DataFrames
	data_long = pd.concat(df_list)
	# Split the Endpoint and Method for easier plotting
	data_long['Endpoint'], data_long['Method'] = zip(*data_long['Endpoint_Method'].apply(lambda x: x.split('_')))
	# Plotting
	fig1, ax1 = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
	ordered_endpoints = ["LogP", "Permeability", "Solubility", "LogD", "hERG"]
	methods = data_long['Method'].unique()
	# Assigning custom colors for specific methods
	colors = plt.cm.PuOr(np.linspace(0.3, 1, len(methods)))  # Colors based on methods
	custom_colors = dict(zip(methods, colors))
	custom_colors['GNN'] = 'darkgrey'
	color_idx = 0
	method_color_dict = {}
	for method in methods:
		if method in custom_colors:
			method_color_dict[method] = custom_colors[method]
		else:
			method_color_dict[method] = colors[color_idx]
			color_idx += 1
	# Calculate the number of methods to determine spacing
	num_methods = len(methods)
	method_spacing = 0.15  # adjust spacing as needed
	bar_width = 0.15  # adjust width of each bar as needed
	import scipy.stats as stats
	# Assuming a 95% confidence level
	confidence_level = 0.95
	for idx, endpoint in enumerate(ordered_endpoints):
		endpoint_data = data_long[data_long['Endpoint'] == endpoint]
		endpoint_data["Value"] *= 100
		base_position = idx - (num_methods * method_spacing / 2)
		for method_idx, method in enumerate(methods):
			method_data = endpoint_data[endpoint_data['Method'] == method]['Value']
			position = base_position + method_idx * method_spacing
			# Calculate mean
			mean = method_data.mean()
			# Calculate standard error
			se = method_data.std() / np.sqrt(len(method_data))
			# Calculate the margin of error for 95% confidence interval
			t_score = stats.t.ppf((1 + confidence_level) / 2, df=len(method_data) - 1)
			margin_of_error = t_score * se
			# Plot the bar
			ax1.bar(position, mean, align='center', alpha=1, 
					color=method_color_dict[method], width=bar_width, label=method, edgecolor='black', linewidth=0.5)
			# Add error bars with confidence interval
			ax1.errorbar(position, mean, yerr=margin_of_error, fmt='none', ecolor='black', capsize=2, linewidth=.7)
	# add_significance_markers(ax1, data_long, ordered_endpoints, methods, bar_width, method_spacing)
	# Customizing the plot
	ax1.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax1.set_axisbelow(True)
	ax1.set_xticks(range(len(ordered_endpoints)))
	ax1.set_xticklabels(ordered_endpoints, fontsize=12)
	ax1.set_xlabel('ADMET Endpoint', fontsize=14)
	ax1.set_ylabel('Success Rate (%)', fontsize=14)
	ax1.axhline(y=33, linewidth=1, color="k", linestyle="--", zorder=-100)
	ax1.set_axisbelow(True)
	# Create a legend for the methods
	handles, labels = [], []
	for method, color in method_color_dict.items():
		handles.append(plt.Rectangle((0,0),1,1, color=color))
		labels.append(method)
	ax1.legend(handles, labels, loc=(0.65, 0.65), framealpha=1, fontsize=12, title_fontsize=14)
	plt.ylim((0,100))
	plt.tight_layout()
	# Save the plot as SVG
	plt.savefig("./figure/Figure_4a.png", format='svg')
	# Save the plot as PNG
	plt.savefig("./figure/Figure_4a.svg", format='png')
	plt.show()

def plot_aggregation_ADMET(ax, csv_file_path, name_agg, admet, color, title):
	df_t = pd.read_csv(csv_file_path)
	df_t=df_t[df_t["Chemist Group"]==3]
	df_t['Mean_SR'] *= 100
	df_t['25th_Percentile_SR'] *= 100
	df_t['75th_Percentile_SR'] *= 100
	df_t['25th_Percentile_SR'] = df_t['Mean_SR'] -  abs(df_t['25th_Percentile_SR'] - df_t['Mean_SR'])/2
	df_t['75th_Percentile_SR'] = df_t['Mean_SR'] +  abs(df_t['25th_Percentile_SR'] - df_t['Mean_SR'])/2
	df_t['25th_Percentile_SR'].fillna(df_t['Mean_SR'], inplace=True)
	df_t['75th_Percentile_SR'].fillna(df_t['Mean_SR'], inplace=True)

	subset = df_t.drop_duplicates("Key")
	keys_with_zero_n = np.insert([ik+1 for ik in range(len(subset))], 0, 0)
	smoothed_data = scipy.ndimage.filters.gaussian_filter1d(subset['Mean_SR'], sigma=1)
	smoothed_data = np.insert(smoothed_data, 0, subset["Mean_SR"].tolist()[0])
	ax.plot(keys_with_zero_n, smoothed_data, label=admet, linewidth=2, color=color)

	smoothed_data_up = scipy.ndimage.filters.gaussian_filter1d(subset['75th_Percentile_SR'], sigma=1)
	smoothed_data_up = np.insert(smoothed_data_up, 0, subset["Mean_SR"].tolist()[0])
	ax.plot(keys_with_zero_n, smoothed_data_up, linewidth=1, color=color, linestyle='--')

	smoothed_data_down = scipy.ndimage.filters.gaussian_filter1d(subset['25th_Percentile_SR'], sigma=1)
	smoothed_data_down = np.insert(smoothed_data_down, 0, subset["Mean_SR"].tolist()[0])
	ax.plot(keys_with_zero_n, smoothed_data_down, linewidth=1, color=color, linestyle='--')

	ax.fill_between(keys_with_zero_n, smoothed_data_up, smoothed_data_down, color=color, alpha=0.1)
	
	d_a = {"most_frequent":"Most Frequent","log_odds":"Log Odds","weighted_by_certitude":"Weighted By Certitude","weighted_by_certitude_and_expertise":"Weighted By Certitude & Expertise",
	"weighted_by_expertise":"Weighted By Expertise", "fuzzy_logic_aggregation":"Fuzzy Logic Aggregation"}

	# Set y-axis label only for the first column
	ax.set_title(title, fontsize=14)

	ax.set_xlabel('Number of participants', fontsize=14)
	ax.set_ylabel('Success Rate (%)', fontsize=14)
	ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
	ax.set_axisbelow(True)
	


def plot_certitude_distribution(df, output_file_path):
	# Convert Certitude to string for grouping
	df["Certitude"] = df["Certitude"].astype(str)

	# Group data for bubble chart
	bubble_data = df.groupby(['Chemist Level', 'Certitude']).size().reset_index(name='Counts')

	# Create a figure for the plot
	fig, ax = plt.subplots(figsize=(6, 5))

	# Determine color scale based on counts
	cmin = bubble_data['Counts'].min()
	cmax = bubble_data['Counts'].max()
	colors = plt.cm.viridis([math.log10(i) * 0.3 for i in bubble_data['Counts']])

	# Create scatter plot
	scatter = ax.scatter(bubble_data['Chemist Level'], bubble_data['Certitude'], 
						 s=bubble_data['Counts'] * 2, alpha=1, c=colors, 
						 edgecolors='k', linewidth=1)

	# Set labels and title
	ax.set_xlabel('Chemist Level', fontsize=14)
	ax.set_ylabel('Certitude', fontsize=14)
	ax.set_title('Distribution of Certitude per level', fontsize=16)

	# Create color bar
	cbar = plt.colorbar(scatter, ax=ax, boundaries=np.linspace(0, 1, 6))
	cbar.set_label('Counts')
	cbar.set_ticks(np.linspace(0, 1, 6))
	cbar.set_ticklabels(np.linspace(cmin, cmax, 6, dtype=int))

	# Show and save the plot
	plt.show()
	fig.savefig(output_file_path, dpi=300, bbox_inches='tight')



def plot_tsne_per_endpoint(df, sr_column='SR', endpoint_column='Endpoint',
						   tsne_columns=['TSNE_PC1', 'TSNE_PC2']):
	"""
	Plots t-SNE scatter plots per endpoint with success rates as color.

	Parameters:
	- df: DataFrame containing the data.
	- sr_column: Column name for success rates.
	- endpoint_column: Column name for endpoint labels.
	- tsne_columns: List of column names for t-SNE components.
	"""
	endpoints = df[endpoint_column].unique()

	for endpoint in endpoints:
		endpoint_data = df[df[endpoint_column] == endpoint]
		endpoint_data[sr_column] *= 100

		norm = Normalize(vmin=0, vmax=80)
		fig, ax = plt.subplots(figsize=(6.2, 5))

		scatter = ax.scatter(
			endpoint_data[tsne_columns[0]], endpoint_data[tsne_columns[1]],
			c=endpoint_data[sr_column], s=30, cmap='RdYlGn', alpha=1, 
			edgecolors='k', linewidth=1, norm=norm)

		ax.set_xlabel('Principal Component 1', fontsize=14)
		ax.set_ylabel('Principal Component 2', fontsize=14)

		title = "Permeability" if "Papp" in endpoint else (
			"Solubility" if "Sapp" in endpoint else endpoint)
		ax.set_title(title, fontsize=14)

		ax.spines['right'].set_visible(True)
		ax.spines['top'].set_visible(True)

		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_ticks(np.arange(0, 81, 10))
		cbar.set_label('Success Rate (%)', fontsize=14)

		fig.savefig(f"./figure/Figure_S13-{endpoint}.svg", format='svg')
		fig.savefig(f"./figure/Figure_S13-{endpoint}.png", format='png')

		plt.tight_layout()
		plt.show()

def get_ecfp4(smiles):
	"""
	Converts a SMILES string to an ECFP4 fingerprint.

	Parameters:
	- smiles: SMILES string.

	Returns:
	- Numpy array of the ECFP4 fingerprint.
	"""
	mol = Chem.MolFromSmiles(smiles)
	return (AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
			if mol is not None else np.zeros(2048))

def compute_ecfp_tsne(df, smiles_column='total_smiles'):
	"""
	Computes t-SNE reduction for ECFP4 fingerprints and updates the DataFrame.

	Parameters:
	- df: DataFrame containing SMILES strings.
	- smiles_column: Column name for SMILES.

	Returns:
	- DataFrame updated with t-SNE components 'TSNE_PC1' and 'TSNE_PC2'.
	"""
	ecfp_list = df[smiles_column].apply(get_ecfp4)
	ecfp_matrix = np.array(ecfp_list.tolist())
	tsne = TSNE(n_components=2, random_state=42)
	tsne_results = tsne.fit_transform(ecfp_matrix)
	df['TSNE_PC1'] = tsne_results[:, 0]
	df['TSNE_PC2'] = tsne_results[:, 1]
	return df

def compute_ecfp_umap(df, smiles_column='total_smiles'):
	"""
	Computes UMAP reduction for ECFP4 fingerprints and updates the DataFrame.

	Parameters:
	- df: DataFrame containing SMILES strings.
	- smiles_column: Column name for SMILES.

	Returns:
	- DataFrame updated with UMAP components 'TSNE_PC1' and 'TSNE_PC2'.
	"""
	ecfp_list = df[smiles_column].apply(get_ecfp4)
	ecfp_matrix = np.array(ecfp_list.tolist())
	reducer = umap.UMAP(n_components=2, random_state=42)
	umap_results = reducer.fit_transform(ecfp_matrix)
	df['TSNE_PC1'] = umap_results[:, 0]
	df['TSNE_PC2'] = umap_results[:, 1]
	return df





def apply_tsne_transformation(df, replace_dict, perplexity=5, random_state=0):
	df['Result'] = np.where(df['Correct_Answer'] == df['Answer'], 1, 0)

	# Aggregate duplicates
	agg_df = df.groupby(['Chemist', 'Question']).agg({
		'Answer': 'first',  # or some other aggregation
		'Certitude': 'mean'  # average certitude
	}).reset_index()

	# Pivot DataFrame
	answer_df = agg_df.pivot(index='Chemist', columns='Question', values=['Answer', 'Certitude']).reset_index()

	# Replace and Scale
	X = answer_df.iloc[:, 1:].values
	df_X = pd.DataFrame(X).replace(replace_dict)
	scaler = MinMaxScaler()
	df_X_scaled = pd.DataFrame(scaler.fit_transform(df_X.fillna(1)), columns=df_X.columns)

	# Perform UMAP analysis
	reducer = umap.UMAP(n_components=2, random_state=42)
	tnse = reducer.fit_transform(df_X_scaled)

	return tnse



def plot_aggregation_ExpertGroup(csv_file_path, output_file_path, name_agg, admet):
    df_t = pd.read_csv(csv_file_path)
    aggregation_methods = ["most_frequent", "log_odds", "weighted_by_expertise"]

    dt_all = df_t.copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    # print(df_t)
    dt_all['Mean_SR']*=100
    dt_all['25th_Percentile_SR']*=100
    dt_all['75th_Percentile_SR']*=100
    dt_all['25th_Percentile_SR'] = dt_all['Mean_SR'] -  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2
    dt_all['75th_Percentile_SR'] = dt_all['Mean_SR'] +  abs(dt_all['25th_Percentile_SR'] - dt_all['Mean_SR'])/2

    dt_all['25th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
    dt_all['75th_Percentile_SR'].fillna(dt_all['Mean_SR'], inplace=True)
    chemist_groups = list(set(list(dt_all["Chemist Group"])))

    for unique_group in chemist_groups:
        # print(unique_group)
        
        group_df = dt_all[dt_all["Chemist Group"] == unique_group]
        group_df = group_df[group_df["Key"]==20]
        #print(group_df["Mean_SR"].tolist())
    
    colors = plt.cm.brg(np.linspace(0., 1, len(chemist_groups)))
    # plt.style.use('ggplot')
    plt.style.use('default')
    # Darken the colors
    def darken_color(color, factor=0.6):
        """Returns a darker shade of the given color."""
        return [max(0, c * factor) for c in color]
    colors =['#94568c', '#65ab7c', '#2976bb']
    colors =['#8e0052', '#276319', '#6f9fc7']

    k = 0
    for group in chemist_groups:
        subset = dt_all[dt_all['Chemist Group'] == group]
        subset = subset.drop_duplicates("Key")
        keys_with_zero_n = np.insert([ik+1 for ik in range(len(subset))], 0, 0)
        label = {1: "Non Expert (1-2)", 2: "Expert (3-5)", 3: "All"}.get(group)
        smoothed_data = scipy.ndimage.filters.gaussian_filter1d(subset['Mean_SR'], sigma=1)
        smoothed_data = np.insert(smoothed_data, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data, label=label, linewidth=2, color=colors[k])
        smoothed_data_up = scipy.ndimage.filters.gaussian_filter1d(subset['75th_Percentile_SR'], sigma=1)
        smoothed_data_up = np.insert(smoothed_data_up, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data_up, linewidth=1, color=colors[k], linestyle='--')
        smoothed_data_down = scipy.ndimage.filters.gaussian_filter1d(subset['25th_Percentile_SR'], sigma=1)
        smoothed_data_down = np.insert(smoothed_data_down, 0, subset["Mean_SR"].tolist()[0])
        ax.plot(keys_with_zero_n, smoothed_data_down, linewidth=1, color=colors[k], linestyle='--')
        ax.fill_between(keys_with_zero_n, smoothed_data_up, smoothed_data_down, color=colors[k], alpha=0.1)
        k += 1

    ax.legend(fontsize=10,loc='lower right',) # bbox_to_anchor=(0.45, .35),framealpha=1)
    
    #ax.set_title(name_agg, fontsize=14)
    ax.set_xlabel('Number of participants', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.grid(axis='y', linestyle='--', color='silver', alpha=1)
    plt.xlim((-0, 92))
    ax.set_axisbelow(True)
    plt.ylim((20, 100))
    plt.tight_layout()
    plt.show()
    fig.savefig(output_file_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_file_path.replace(".png",".svg"), dpi=300, bbox_inches='tight')


def _load_responses():
    CI_Input = PandasTools.LoadSDF("./data/CollectiveIntelligence/CI_Answer_v3-Structures_and_experimental_data.sdf")
    CI_Input["total_smiles"] = [Chem.MolToSmiles(i) for i in CI_Input["ROMol"].tolist()]
    CI_Input_Frequent = pd.read_csv("./data/CollectiveIntelligence/CI_Answer_v3-Response_Most_Frequent.csv")
    CI_Input["Slide_ID"] = CI_Input["Slide_ID"].astype(str)
    CI_Input_Frequent["Slide_ID"] = CI_Input_Frequent["Slide_ID"].astype(str)
    merged_df = CI_Input.merge(CI_Input_Frequent, on='Slide_ID')

    df_A = pd.read_csv('./data/CollectiveIntelligence/CI_Answer_A.csv', sep=',')
    df_B = pd.read_csv('./data/CollectiveIntelligence/CI_Answer_B.csv', sep=',')

    df_A['Result'] = np.where(df_A['Correct_Answer'] == df_A['Answer'], 1, 0)
    df_B['Result'] = np.where(df_B['Correct_Answer'] == df_B['Answer'], 1, 0)

    df_A_count = analyze_data(df_A)
    df_B_count = analyze_data(df_B)

    df_count = pd.concat([df_A_count,df_B_count])
    df_count = df_count.rename(columns = {"Success_Rate":"SR"})
    df_count_sorted = df_count.sort_values("SR", ascending = True)
    merged_df["Slide_ID"] = [str(int(i)) for i in merged_df["Slide_ID"].tolist()]
    Merged_Slide_to_SR = df_count_sorted.merge(merged_df, on='Slide_ID')

    # Example usage
    result_df_SLIDE_to_SR = compute_ecfp_tsne(Merged_Slide_to_SR)
    endpoint_data = result_df_SLIDE_to_SR.copy()

    result_df_SLIDE_to_SR['Answer_Correct'] = (result_df_SLIDE_to_SR['Most_Given_Answer'] == result_df_SLIDE_to_SR['ID_Reponse']).astype(int)
    return(Merged_Slide_to_SR, CI_Input, endpoint_data, result_df_SLIDE_to_SR)
    
    
__all__ = ['apply_tsne_transformation',
 'compute_ecfp_tsne',
 'compute_ecfp_umap',
 'get_ecfp4',
 'plot_tsne_per_endpoint',
 '_UMAP_ADMET',
 '_UMAP_SR',
 '_bubble_ADMET',
 '_distribution_ADMET',
 '_distribution_groups',
 '_generatre_benchmark',
 '_waffle_plot',
 'darken_color',
 'plot_aggregation_ADMET',
 'plot_aggregation_S7',
 'plot_certitude_distribution',
 'plot_certitude_success_distribution_color',
 'plot_distribution',
 'plot_distribution_of_scores_S1B',
 'plot_distribution_of_scores_fig1c',
 'plot_hexbin',
 'plot_success_rate_by_endpoint',
 'rgb_to_hex',
 '_load_responses',
 '_merge_data_CI',
 'assign_chemist_group_S1A',
 'assign_chemist_group_spe_S1B',
 'compute_most_frequent',
 'compute_most_frequent_combined_weight',
 'plot_aggregation_AM',
 'plot_aggregation_expert_group',
 'plot_distribution_of_scores',
 'plot_distribution_of_scores_session',
 'transform_dataset_B_v2',
 'analyze_data',
 'assign_chemist_group',
 'columns_to_dict',
 'compute_most_frequent_Endpoint',
 'compute_most_frequent_combined_weight_Endpoint',
 'compute_scores',
 'reclassify_chemist_by_sr',
 'remove_consistent_chemists',
 'plot_aggregation_ExpertGroup']