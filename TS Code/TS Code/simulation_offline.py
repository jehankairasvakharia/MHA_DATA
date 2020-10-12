import pandas as pd
from .policies_offline import thompson_sampling
from .assign_prob import compute_assign_prob_TS_2
from .simulate_and_plot import plot_beta_dist, plot_assignment_prob, plot_success_failure

if __name__ == "__main__":
	data_path = 'design_3_input.xlsx'
	context = {'policy_parameters':
				{'outcome_variable_name': 'value', 'max_rating': 4, 'prior':{'success': 1, 'failure': 1}},
			'mooclet': 156, 'versions': [5302, 5303, 5304]}
	factor = 'design3'
	out_folder = './output/{}_offline/'.format(factor)
	csv_path = '{}{}_models_offline.csv'.format(out_folder, factor)
	stops = [10, 50, 119]
	num_sims = 1000
	levels = ['banality', 'question', 'question+quote'] # description of versions

	# Read csv and extract TS data
	data = pd.read_excel(data_path)
	data = data.dropna(subset=['version_id_{}'.format(context['mooclet'])])
	data = data[data['policy_id_{}'.format(context['mooclet'])] == 3].sort_values('timestamp_{}'.format(context['mooclet']), ignore_index=True)

	# Increment the number of data points one by one and save alpha and beta to a DataFrame
	columns = []
	for version in context['versions']:
		columns.append('successes_{}'.format(version))
		columns.append('failures_{}'.format(version))
		columns.append('probability_{}'.format(version))
	df = pd.DataFrame(columns=columns)
	for i in range(data.shape[0]):
		simulation_data = data[:i]
		alpha_beta = thompson_sampling([], context, simulation_data)
		alphas = []
		betas = []
		for version in context['versions']:
			alphas.append(alpha_beta['successes_{}'.format(version)])
			betas.append(alpha_beta['failures_{}'.format(version)])
		alpha_beta.update(compute_assign_prob_TS_2(alphas, betas, num_sims, context['versions']))
		df = df.append(alpha_beta, ignore_index=True)
	df = df.fillna(0)

	# Save the results
	df.to_csv(csv_path)

	# Plot
	ap_lists = []
	for version in context['versions']:
		ap_lists.append(df['probability_{}'.format(version)])
	for stop in stops:
		plot_assignment_prob(ap_lists, levels, stop, factor, out_folder)
		plot_success_failure(df, context['versions'], levels, stop, factor, out_folder)
		for i in range(len(context['versions'])):
			plot_beta_dist(df['successes_{}'.format(context['versions'][i])], df['failures_{}'.format(context['versions'][i])],
				stop, levels[i], out_folder)
