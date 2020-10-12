import pandas as pd
from policies_offline import thompson_sampling
from scipy.stats import beta
import numpy as np
from assign_prob import *
import matplotlib.pyplot as plt

def plot_beta_dist(alphas: pd.Series, betas: pd.Series, stop: int, level: str, out_folder: str):
    for i in range(stop):
        alpha_curr = alphas[i]
        beta_curr = betas[i]
        x = np.arange (0.01, 1, 0.01)
        plt.plot(x, beta.pdf(x, alpha_curr, beta_curr))
    plt.title("Beta Distribution for First {} Participants For {}".format(stop, level))
    plt.savefig("{}{}_{}.png".format(out_folder, level, stop))
    plt.close()

def plot_assignment_prob(ap_lists: list, levels: list, stop: int, factor: str, out_folder: str):
    for i in range(len(ap_lists)):
        plt.plot(ap_lists[i][:stop], ".-", label = levels[i])
    plt.legend()
    plt.xlabel("participant index")
    plt.title("Assignment Probability for {} Factor For First {} Participants".format(factor, stop))
    plt.ylabel("assignment probability")
    plt.savefig("{}assignment_probability_{}_{}.png".format(out_folder, factor, stop))
    plt.close()

def plot_success_failure(df: pd.DataFrame, versions: list, levels: list, stop: int, factor: str, out_folder: str):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] # if you have more levels, add on your own
    for i in range(len(versions)):
        plt.plot(df["failures_{}".format(versions[i])][0:stop], color=colors[i], marker="o", label="{} failures".format(levels[i]))
        plt.plot(df["successes_{}".format(versions[i])][0:stop], color=colors[i], marker= "*", label="{} successes".format(levels[i]))
    plt.ylabel("Count")
    plt.title("Number of Successes and Failures {} Factor For First {} Participants".format(factor, stop))
    plt.xlabel("participant index")
    plt.legend()
    plt.savefig("{}success_failure_{}_{}.png".format(out_folder, factor, stop))
    plt.close()

if __name__ == "__main__":
    data_path = './data/explanation_helpful_mooclet.csv'
    data_exp_path = "./data/Jacob_exp_no_consent_preprocessed.csv"
    

    
    should_be_dropped_exp = ['mooclets_working_explanation', 'earliest_mooclets_explanation']
    should_be_dropped_exp = ['mooclets_working_explanation']
    data_exp = pd.read_csv(data_exp_path)
    
    for column in should_be_dropped_exp:
        data_exp = data_exp[data_exp[column] == 1]
    
    data_exp_ts = data_exp[data_exp['policy_explanation_TS'] == 1]
    preview_ids = data_exp_ts[data_exp_ts['DistributionChannel'] == 'preview']["StudentRandomID"]
    
    context = {'policy_parameters':
				{'outcome_variable_name': 'explanation_helpful', 'max_rating': 1, 'prior':{'success': 1, 'failure': 1}}, 
			'mooclet': 123, 'versions': [5134, 5135]} #5314 is short, 5135 is long
    out_path = './output/explanation_helpful_models.csv'

	# Read csv and extract TS data
    data = pd.read_csv(data_path)
    data = data[data['policy_id_'.format(context['mooclet'])] == 3].sort_values('timestamp_explanation', ignore_index=True)

	# Increment the number of data points one by one and save alpha and beta to a DataFrame
    columns = [] #alpha is successes, beta is failures
    for version in context['versions']:
        columns.append('successes_{}'.format(version))
        columns.append('failures_{}'.format(version))
    df = pd.DataFrame(columns=columns)
    for i in range(data.shape[0]):
        simulation_data = data[:i+1]
        alpha_beta = thompson_sampling([], context, simulation_data)
        df = df.append(alpha_beta, ignore_index=True)

	# Save the results
    df.to_csv(out_path)
    
    #posteriors
    arm_1_params = df[["successes_5134", "failures_5134"]]
    arm_2_params = df[["successes_5135", "failures_5135"]]
    stop = 10#must be less than 600
    
    for i in range(stop):  
        alpha_curr = arm_1_params["successes_5134"][i]
        beta_curr = arm_1_params["failures_5134"][i]
      #  print(alpha_curr, beta_curr)
        x = np.arange (0.01, 1, 0.01)
        plt.plot(x, beta.pdf(x, alpha_curr, beta_curr))
    plt.title("Beta Distribution for First 10 Students For Short Explanation")
    plt.show()
    plt.close()
   
    for i in range(stop):  
        alpha_curr = arm_2_params["successes_5135"][i]
        beta_curr = arm_2_params["failures_5135"][i]
       # print(alpha_curr, beta_curr)
        x = np.arange (0.01, 1, 0.01)
        plt.plot(x, beta.pdf(x, alpha_curr, beta_curr))
    plt.title("Beta Distribution for First 10 Students For Long Explanation")
    plt.show()
    plt.close()
    ap_list_1 = []
    ap_list_2 = []
    for i in range(stop):
        alpha_curr_1 = arm_1_params["successes_5134"][i]
        beta_curr_1 = arm_1_params["failures_5134"][i]
        alpha_curr_2 = arm_2_params["successes_5135"][i]
        beta_curr_2 = arm_2_params["failures_5135"][i]
        
        ap = compute_assign_prob_TS(alpha_curr_1, beta_curr_1, alpha_curr_2, beta_curr_2, 1000)
        ap_list_1.append(ap[0])
        ap_list_2.append(ap[1])
    plt.plot(ap_list_1, ".-", label = "short explanation")
    plt.plot(ap_list_2, ".-", label = "long explanation")
    plt.legend()
    plt.xlabel("student index")
    plt.title("Assignment Probability for Explanation Factor For First 10 Students")
    plt.ylabel("assignment probability")
    plt.show()
    plt.close()
        
    
    plt.plot(arm_1_params["failures_5134"][0:stop], color = "blue", marker="o", label="short explanation failures")
    plt.plot(arm_1_params["successes_5134"][0:stop], color = "blue", marker= "*", label="short explanation successes")
    plt.plot(arm_2_params["failures_5135"][0:stop], color = "red", marker="o", label="long explanation failures")
    plt.plot(arm_2_params["successes_5135"][0:stop], color = "red", marker= "*", label="long explanation successes")
    plt.ylabel("Count")
    plt.title("Number of Successes and Failures Explanation Factor For First 10 Students")
    plt.xlabel("student index")
    plt.legend()